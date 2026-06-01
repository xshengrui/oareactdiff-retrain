import argparse
import pickle
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from oa_reactdiff.dataset.transition1x import ProcessedTS1x


DEFAULT_DATASET_PATH = PROJECT_ROOT / "oa_reactdiff" / "data" / "transition1x" / "valid_addprop.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xyz_from_ckpt"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "oa_reactdiff" / "trainer" / "our_new_pretrained-ts1x-diff.ckpt"
)
RAW_TAR_CONVERTER_VERSION = 3
ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}
ATOMIC_NUMBER_TO_COVALENT_RADIUS = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export reactant/product xyz files and repeated TS predictions from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        type=str,
        help="Path to the trained .ckpt file.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_DATASET_PATH),
        type=str,
        help="Path to the dataset .pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        type=str,
        help="Directory for exported xyz files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help='Device to use: "auto", "cuda", or "cpu".',
    )
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size for inference.")
    parser.add_argument("--timesteps", default=250, type=int, help="Diffusion timesteps.")
    parser.add_argument(
        "--noise-schedule",
        default="cosine",
        type=str,
        help="Noise schedule used for inference. Defaults to cosine to match this checkpoint.",
    )
    parser.add_argument("--resamplings", default=5, type=int, help="RePaint resamplings.")
    parser.add_argument("--jump-length", default=5, type=int, help="RePaint jump length.")
    parser.add_argument("--repeats", default=30, type=int, help="Number of TS predictions per sample.")
    parser.add_argument(
        "--single-frag-only",
        default=1,
        type=int,
        help=(
            "Whether to keep only single-fragment reactions (1 or 0). "
            "Default 1 matches this checkpoint's training config."
        ),
    )
    parser.add_argument(
        "--use-by-ind",
        default=1,
        type=int,
        help="Whether to filter by the dataset use_ind split (1 or 0).",
    )
    parser.add_argument(
        "--position-key",
        default="positions",
        type=str,
        help="Position key inside the dataset file.",
    )
    parser.add_argument(
        "--max-samples",
        default=-1,
        type=int,
        help="Limit the number of exported samples. Use -1 for all samples.",
    )
    parser.add_argument(
        "--max-atoms",
        default=-1,
        type=int,
        help=(
            "Keep only reactions with at most this many atoms. "
            "Use -1 to disable. The TS1x checkpoint was trained on <=23 atoms."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--save-true",
        default=1,
        type=int,
        help="Whether to also save ground-truth r/ts/p into true_rts_p.xyz (1 or 0).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only convert/validate the dataset input and exit without loading the model.",
    )
    parser.add_argument(
        "--stop-on-nan",
        action="store_true",
        help="Stop inference and print batch/sample diagnostics if generated tensors contain NaN/Inf.",
    )
    parser.add_argument(
        "--log-batches",
        action="store_true",
        help="Print rxn ids and geometry diagnostics before each inference batch.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def parse_xyz_bytes(content: bytes, member_name: str):
    lines = content.decode("utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty xyz file: {member_name}")

    natoms = int(lines[0].strip())
    atom_lines = [line.strip() for line in lines[2 : 2 + natoms] if line.strip()]
    if len(atom_lines) != natoms:
        raise ValueError(
            f"Expected {natoms} atoms in {member_name}, found {len(atom_lines)}"
        )

    charges = []
    positions = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed atom line in {member_name}: {line}")
        element = fields[0]
        if element not in ELEMENT_TO_ATOMIC_NUMBER:
            raise ValueError(f"Unsupported element {element!r} in {member_name}")
        charges.append(ELEMENT_TO_ATOMIC_NUMBER[element])
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return natoms, charges, np.asarray(positions, dtype=np.float32)


def empty_fragment_dataset():
    return {
        "num_atoms": [],
        "charges": [],
        "fragments": [],
        "positions": [],
        "rxn": [],
    }


def infer_connected_fragments(charges, positions, bond_scale=1.25, bond_slack=0.25):
    natoms = len(charges)
    parents = list(range(natoms))

    def find(atom_index):
        while parents[atom_index] != atom_index:
            parents[atom_index] = parents[parents[atom_index]]
            atom_index = parents[atom_index]
        return atom_index

    def union(atom_i, atom_j):
        root_i = find(atom_i)
        root_j = find(atom_j)
        if root_i != root_j:
            parents[root_j] = root_i

    for atom_i in range(natoms):
        for atom_j in range(atom_i + 1, natoms):
            radius_i = ATOMIC_NUMBER_TO_COVALENT_RADIUS[charges[atom_i]]
            radius_j = ATOMIC_NUMBER_TO_COVALENT_RADIUS[charges[atom_j]]
            cutoff = bond_scale * (radius_i + radius_j) + bond_slack
            distance = np.linalg.norm(positions[atom_i] - positions[atom_j])
            if distance <= cutoff:
                union(atom_i, atom_j)

    fragments_by_root = {}
    for atom_index in range(natoms):
        fragments_by_root.setdefault(find(atom_index), []).append(atom_index)
    return list(fragments_by_root.values())


def build_dataset_from_raw_tar(tar_path: Path):
    species_to_file = {
        "reactant": "R.xyz",
        "transition_state": "TS.xyz",
        "product": "P.xyz",
    }
    records = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[-1] not in species_to_file.values():
                continue
            reaction_id = parts[-2]
            records.setdefault(reaction_id, {})
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            records[reaction_id][parts[-1]] = parse_xyz_bytes(
                extracted.read(),
                member.name,
            )

    complete_reactions = sorted(
        reaction_id
        for reaction_id, files in records.items()
        if all(filename in files for filename in species_to_file.values())
    )
    if not complete_reactions:
        raise ValueError(f"No complete R.xyz/P.xyz/TS.xyz reactions found in {tar_path}")

    dataset = {
        "reactant": empty_fragment_dataset(),
        "transition_state": empty_fragment_dataset(),
        "product": empty_fragment_dataset(),
        "single_fragment": [],
        "use_ind": list(range(len(complete_reactions))),
        "raw_tar_converter_version": RAW_TAR_CONVERTER_VERSION,
    }

    for reaction_id in complete_reactions:
        reference_charges = None
        reference_natoms = None
        reaction_fragments = {}
        for species, filename in species_to_file.items():
            natoms, charges, positions = records[reaction_id][filename]
            if reference_natoms is None:
                reference_natoms = natoms
                reference_charges = charges
            elif natoms != reference_natoms or charges != reference_charges:
                raise ValueError(
                    f"Inconsistent atom order/count for {reaction_id}: {filename}"
                )

            fragments = infer_connected_fragments(charges, positions)
            reaction_fragments[species] = fragments
            dataset[species]["num_atoms"].append(natoms)
            dataset[species]["charges"].append(charges)
            dataset[species]["fragments"].append(fragments)
            dataset[species]["positions"].append(positions)
            dataset[species]["rxn"].append(reaction_id)
        dataset["single_fragment"].append(
            int(all(len(fragments) == 1 for fragments in reaction_fragments.values()))
        )

    return dataset


def ensure_model_input_dataset(dataset_path: Path, output_dir: Path):
    suffixes = "".join(dataset_path.suffixes)
    if suffixes.endswith(".tar.gz"):
        processed_dir = output_dir / "_processed_inputs"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / f"{dataset_path.name[:-7]}.pkl"
        needs_conversion = (
            not processed_path.exists()
            or processed_path.stat().st_mtime < dataset_path.stat().st_mtime
        )
        if not needs_conversion:
            cached_dataset = load_pickle(processed_path)
            needs_conversion = (
                cached_dataset.get("raw_tar_converter_version")
                != RAW_TAR_CONVERTER_VERSION
            )
        if needs_conversion:
            print(f"converting raw tar dataset to model input pkl: {processed_path}")
            dataset = build_dataset_from_raw_tar(dataset_path)
            with open(processed_path, "wb") as handle:
                pickle.dump(dataset, handle)
            print(
                f"converted_reactions={len(dataset['single_fragment'])} "
                f"source={dataset_path}"
            )
        return processed_path
    return dataset_path


def validate_model_input_dataset(dataset_path: Path):
    dataset = load_pickle(dataset_path)
    required_species = ["reactant", "transition_state", "product"]
    required_keys = ["num_atoms", "charges", "fragments", "positions", "rxn"]

    sample_count = len(dataset["single_fragment"])
    issues = []
    max_abs_coord = 0.0
    max_span = 0.0
    atom_counts = []
    atomic_numbers = set()
    fragment_counts = {species: [] for species in required_species}

    for species in required_species:
        if species not in dataset:
            issues.append(f"missing species: {species}")
            continue
        for key in required_keys:
            if key not in dataset[species]:
                issues.append(f"missing key: {species}.{key}")
                continue
            if len(dataset[species][key]) != sample_count:
                issues.append(
                    f"length mismatch: {species}.{key} has "
                    f"{len(dataset[species][key])}, expected {sample_count}"
                )

    for idx in range(sample_count):
        ref_charges = None
        ref_natoms = None
        for species in required_species:
            natoms = int(dataset[species]["num_atoms"][idx])
            charges = list(dataset[species]["charges"][idx])[:natoms]
            positions = np.asarray(dataset[species]["positions"][idx])[:natoms]
            fragments = dataset[species]["fragments"][idx]
            if isinstance(fragments, list):
                fragment_counts[species].append(len(fragments))

            if positions.shape != (natoms, 3):
                issues.append(
                    f"bad position shape at sample={idx} species={species}: "
                    f"{positions.shape}, natoms={natoms}"
                )
            if not np.isfinite(positions).all():
                issues.append(f"non-finite positions at sample={idx} species={species}")
            if len(charges) != natoms:
                issues.append(
                    f"charge length mismatch at sample={idx} species={species}: "
                    f"{len(charges)}, natoms={natoms}"
                )
            if not isinstance(fragments, list) or not fragments:
                issues.append(f"bad fragments at sample={idx} species={species}")
            else:
                flattened_fragments = [
                    atom_index for fragment in fragments for atom_index in fragment
                ]
                if sorted(flattened_fragments) != list(range(natoms)):
                    issues.append(
                        f"fragments do not cover atoms at sample={idx} "
                        f"species={species}"
                    )
            unsupported = sorted(set(charges) - set(ELEMENT_TO_ATOMIC_NUMBER.values()))
            if unsupported:
                issues.append(
                    f"unsupported atomic numbers at sample={idx} "
                    f"species={species}: {unsupported}"
                )

            if ref_natoms is None:
                ref_natoms = natoms
                ref_charges = charges
            elif natoms != ref_natoms or charges != ref_charges:
                issues.append(
                    f"R/TS/P atom order mismatch at sample={idx} species={species}"
                )

            max_abs_coord = max(max_abs_coord, float(np.max(np.abs(positions))))
            max_span = max(max_span, float(np.ptp(positions, axis=0).max()))
            atomic_numbers.update(charges)

        atom_counts.append(ref_natoms)

    print(f"validated_dataset={dataset_path}")
    print(f"samples={sample_count}")
    if atom_counts:
        print(
            "natoms min/median/max="
            f"{min(atom_counts)}/{np.median(atom_counts):.1f}/{max(atom_counts)}"
        )
    print(f"atomic_numbers={sorted(atomic_numbers)}")
    print(
        "single_fragment counts="
        f"{dict((int(v), int(c)) for v, c in zip(*np.unique(dataset['single_fragment'], return_counts=True)))}"
    )
    for species in required_species:
        if fragment_counts[species]:
            values, counts = np.unique(fragment_counts[species], return_counts=True)
            print(
                f"{species}_fragment_counts="
                f"{dict((int(v), int(c)) for v, c in zip(values, counts))}"
            )
    print(f"max_abs_coord={max_abs_coord:.6g} max_molecular_span={max_span:.6g}")
    print(f"issues={len(issues)}")
    for issue in issues[:20]:
        print(f"issue: {issue}")
    if issues:
        raise ValueError("Dataset validation failed.")


def filter_dataset_by_max_atoms(raw_dataset, max_atoms: int):
    if max_atoms <= 0:
        return raw_dataset

    keep_indices = [
        idx
        for idx, natoms in enumerate(raw_dataset["reactant"]["num_atoms"])
        if int(natoms) <= max_atoms
    ]
    dropped = len(raw_dataset["single_fragment"]) - len(keep_indices)
    if dropped == 0:
        print(f"max_atoms_filter={max_atoms} kept all {len(keep_indices)} samples")
        return raw_dataset

    filtered = {}
    for key, value in raw_dataset.items():
        if key in ["reactant", "transition_state", "product"]:
            filtered[key] = {
                sub_key: [sub_value[idx] for idx in keep_indices]
                for sub_key, sub_value in value.items()
            }
        elif key == "single_fragment":
            filtered[key] = [value[idx] for idx in keep_indices]
        elif key == "use_ind":
            filtered[key] = list(range(len(keep_indices)))
        else:
            filtered[key] = value

    print(
        f"max_atoms_filter={max_atoms} kept={len(keep_indices)} "
        f"dropped={dropped}"
    )
    if not keep_indices:
        raise ValueError(
            f"No samples remain after --max-atoms {max_atoms}. "
            "This checkpoint was trained on smaller TS1x systems; use a checkpoint "
            "trained for larger molecules or disable the filter and expect NaN risk."
        )
    return filtered


def write_filtered_dataset(raw_dataset, source_path: Path, output_dir: Path, max_atoms: int):
    filtered = filter_dataset_by_max_atoms(raw_dataset, max_atoms)
    if filtered is raw_dataset:
        return source_path, raw_dataset

    filtered_dir = output_dir / "_filtered_inputs"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = filtered_dir / f"{source_path.stem}_max_atoms_{max_atoms}.pkl"
    with open(filtered_path, "wb") as handle:
        pickle.dump(filtered, handle)
    return filtered_path, filtered


def summarize_sample_geometry(dataset, dataset_index: int):
    pieces = []
    for species in ["reactant", "transition_state", "product"]:
        data = dataset.raw_dataset[species]
        natoms = int(data["num_atoms"][dataset_index])
        positions = np.asarray(data["positions"][dataset_index])[:natoms]
        centroid = positions.mean(axis=0)
        centered = positions - centroid
        if natoms > 1:
            pairwise = positions[:, None, :] - positions[None, :, :]
            distances = np.sqrt(np.sum(pairwise * pairwise, axis=-1))
            distances[distances == 0] = np.inf
            min_dist = float(np.min(distances))
        else:
            min_dist = float("nan")
        pieces.append(
            f"{species}: natoms={natoms} min_dist={min_dist:.4f} "
            f"max_abs_centered={float(np.max(np.abs(centered))):.4f} "
            f"span={float(np.ptp(positions, axis=0).max()):.4f}"
        )
    return " | ".join(pieces)


def compute_selected_indices(raw_dataset, single_frag_only: bool, use_by_ind: bool):
    if single_frag_only:
        single_frag_inds = np.where(np.array(raw_dataset["single_fragment"]) == 1)[0]
    else:
        single_frag_inds = np.array(range(len(raw_dataset["single_fragment"])))

    if use_by_ind:
        use_inds = raw_dataset["use_ind"]
    else:
        use_inds = range(len(raw_dataset["single_fragment"]))

    return list(set(single_frag_inds).intersection(set(use_inds)))


def split_by_sample(fragment_tensors, fragments_nodes):
    split_points = torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    return [torch.tensor_split(fragment_tensor, split_points) for fragment_tensor in fragment_tensors]


def write_xyz_block(handle, sample_tensor):
    natoms = int(sample_tensor.shape[0])
    handle.write(f"{natoms}\n\n")
    for row in sample_tensor:
        coord = row[:3].cpu().numpy()
        atomic_number = int(row[-1].long().item())
        if atomic_number == 1:
            element = "H"
        elif atomic_number == 6:
            element = "C"
        elif atomic_number == 7:
            element = "N"
        elif atomic_number == 8:
            element = "O"
        elif atomic_number == 9:
            element = "F"
        else:
            raise ValueError(f"Unsupported atomic number: {atomic_number}")
        handle.write(f"{element} {coord[0]} {coord[1]} {coord[2]}\n")


def safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_sample_metadata(dataset, dataset_index: int, source_index):
    reactant = dataset.raw_dataset["reactant"]
    product = dataset.raw_dataset["product"]

    metadata = {
        "dataset_index": dataset_index,
        "source_index": source_index,
        "num_atoms": int(safe_value(reactant["num_atoms"][dataset_index])),
    }

    if "rxn" in reactant:
        metadata["reaction"] = safe_value(reactant["rxn"][dataset_index])
    if "formula" in reactant:
        metadata["reactant_formula"] = safe_value(reactant["formula"][dataset_index])
    if "formula" in product:
        metadata["product_formula"] = safe_value(product["formula"][dataset_index])
    if "smi" in reactant:
        metadata["reactant_smi"] = safe_value(reactant["smi"][dataset_index])
    if "smi" in product:
        metadata["product_smi"] = safe_value(product["smi"][dataset_index])

    return metadata


def load_ddpm_from_checkpoint(checkpoint_path: Path, device: torch.device):
    from oa_reactdiff.trainer.pl_trainer import DDPMModule

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    ddpm_trainer = DDPMModule(**checkpoint["hyper_parameters"])
    ddpm_trainer.load_state_dict(checkpoint["state_dict"])
    return ddpm_trainer.to(device)


def set_new_schedule_local(
    ddpm_trainer,
    timesteps: int,
    device: torch.device,
    noise_schedule: str = "polynomial_2",
):
    from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=1e-5,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=ddpm_trainer.ddpm.norm_values,
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    return ddpm_trainer.to(device)


def inpaint_batch_local(
    batch,
    ddpm_trainer,
    resamplings: int,
    jump_length: int,
    frag_fixed=None,
):
    from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING

    if frag_fixed is None:
        frag_fixed = [0, 2]

    representations, conditions = batch
    xh_fixed = [
        torch.cat(
            [representation[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for representation in representations
    ]
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [representation["size"] for representation in representations]
    out_samples, _ = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=resamplings,
        jump_length=jump_length,
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=frag_fixed,
    )
    return out_samples[0], xh_fixed, fragments_nodes


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = ensure_model_input_dataset(dataset_path, output_dir)

    if args.validate_only:
        validate_model_input_dataset(dataset_path)
        return

    device = resolve_device(args.device)
    dataset_device = "cuda" if device.type == "cuda" else "cpu"
    single_frag_only = bool(args.single_frag_only)
    use_by_ind = bool(args.use_by_ind)
    save_true = bool(args.save_true)

    raw_dataset = load_pickle(dataset_path)
    dataset_path, raw_dataset = write_filtered_dataset(
        raw_dataset=raw_dataset,
        source_path=dataset_path,
        output_dir=output_dir,
        max_atoms=args.max_atoms,
    )
    selected_indices = compute_selected_indices(
        raw_dataset=raw_dataset,
        single_frag_only=single_frag_only,
        use_by_ind=use_by_ind,
    )
    rxn_to_source_index = {
        rxn_id: idx for idx, rxn_id in enumerate(raw_dataset["reactant"]["rxn"])
    }

    dataset = ProcessedTS1x(
        npz_path=str(dataset_path),
        center=True,
        pad_fragments=0,
        device=dataset_device,
        zero_charge=False,
        remove_h=False,
        single_frag_only=single_frag_only,
        swapping_react_prod=False,
        use_by_ind=use_by_ind,
        position_key=args.position_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    ddpm_trainer = load_ddpm_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    ddpm_trainer = set_new_schedule_local(
        ddpm_trainer=ddpm_trainer,
        timesteps=args.timesteps,
        device=device,
        noise_schedule=args.noise_schedule,
    )
    ddpm_trainer.eval()

    effective_samples = len(dataset)
    if args.max_samples > 0:
        effective_samples = min(effective_samples, args.max_samples)
    print(
        f"dataset_samples={len(dataset)} raw_samples={len(raw_dataset['single_fragment'])} "
        f"effective_samples={effective_samples} single_frag_only={single_frag_only} use_by_ind={use_by_ind}"
    )

    exported_samples = 0
    mapping_path = output_dir / "sample_to_rxn_mapping.txt"
    mapping_handle = open(mapping_path, "w")
    mapping_handle.write("sample_id\trxn\n")

    with torch.no_grad():
        for repeat_idx in range(args.repeats):
            print(f"[repeat {repeat_idx + 1}/{args.repeats}] starting")
            processed = 0
            sample_offset = 0

            for batch_idx, batch in enumerate(loader):
                if processed >= effective_samples:
                    break

                batch_start = time.time()
                if args.log_batches:
                    current_batch_size = int(batch[0][0]["size"].size(0))
                    batch_indices = [
                        sample_offset + local_debug_idx
                        for local_debug_idx in range(current_batch_size)
                        if sample_offset + local_debug_idx < len(dataset)
                    ]
                    rxn_ids = [
                        safe_value(dataset.raw_dataset["reactant"]["rxn"][dataset_index])
                        for dataset_index in batch_indices
                    ]
                    print(
                        f"[repeat {repeat_idx + 1}/{args.repeats}] "
                        f"batch={batch_idx} starting dataset_indices={batch_indices} "
                        f"rxn_ids={rxn_ids}"
                    )
                    for dataset_index in batch_indices:
                        print(
                            f"debug sample dataset_index={dataset_index} "
                            f"{summarize_sample_geometry(dataset, dataset_index)}"
                        )
                out_samples, xh_fixed, fragments_nodes = inpaint_batch_local(
                    batch=batch,
                    ddpm_trainer=ddpm_trainer,
                    resamplings=args.resamplings,
                    jump_length=args.jump_length,
                    frag_fixed=[0, 2],
                )

                if args.stop_on_nan:
                    tensors_to_check = {
                        "out_reactant": out_samples[0],
                        "out_ts": out_samples[1],
                        "out_product": out_samples[2],
                        "fixed_reactant": xh_fixed[0],
                        "fixed_ts": xh_fixed[1],
                        "fixed_product": xh_fixed[2],
                    }
                    bad_tensors = [
                        name
                        for name, tensor in tensors_to_check.items()
                        if not torch.isfinite(tensor).all()
                    ]
                    if bad_tensors:
                        print(
                            f"nonfinite tensors detected at repeat={repeat_idx} "
                            f"batch={batch_idx}: {bad_tensors}"
                        )
                        current_batch_size = int(fragments_nodes[0].size(0))
                        for local_debug_idx in range(current_batch_size):
                            dataset_index = sample_offset + local_debug_idx
                            if dataset_index >= len(dataset):
                                continue
                            rxn_id = safe_value(
                                dataset.raw_dataset["reactant"]["rxn"][dataset_index]
                            )
                            print(
                                f"debug sample local={local_debug_idx} "
                                f"dataset_index={dataset_index} rxn={rxn_id} "
                                f"{summarize_sample_geometry(dataset, dataset_index)}"
                            )
                        raise RuntimeError("Stopping because generated tensors contain NaN/Inf.")

                split_fixed = split_by_sample(xh_fixed, fragments_nodes)
                split_output = split_by_sample(out_samples, fragments_nodes)
                batch_size = len(split_output[1])
                batch_limit = min(batch_size, effective_samples - processed)

                for local_idx in range(batch_limit):
                    dataset_index = sample_offset + local_idx
                    source_index = None
                    rxn_id = safe_value(dataset.raw_dataset["reactant"]["rxn"][dataset_index])
                    if rxn_id in rxn_to_source_index:
                        source_index = int(rxn_to_source_index[rxn_id])
                    elif dataset_index < len(selected_indices):
                        source_index = int(selected_indices[dataset_index])

                    if source_index is not None:
                        sample_dir_name = f"sample_src_{source_index:05d}"
                    else:
                        sample_dir_name = f"sample_{dataset_index:05d}"
                    sample_dir = output_dir / sample_dir_name
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    if repeat_idx == 0:
                        mapping_handle.write(f"{sample_dir_name}\t{rxn_id}\n")
                        if save_true:
                            true_xyz = sample_dir / "true_rts_p.xyz"
                            with open(true_xyz, "w") as handle:
                                write_xyz_block(handle, split_fixed[0][local_idx])  # reactant
                                write_xyz_block(handle, split_fixed[1][local_idx])  # true TS
                                write_xyz_block(handle, split_fixed[2][local_idx])  # product

                    combined_xyz = sample_dir / f"rts_p_repeat_{repeat_idx:02d}.xyz"
                    with open(combined_xyz, "w") as handle:
                        write_xyz_block(handle, split_fixed[0][local_idx])  # reactant
                        write_xyz_block(handle, split_output[1][local_idx])  # predicted TS
                        write_xyz_block(handle, split_fixed[2][local_idx])  # product

                processed += batch_limit
                sample_offset += batch_size
                elapsed = time.time() - batch_start
                print(
                    f"[repeat {repeat_idx + 1}/{args.repeats}] "
                    f"batch={batch_idx} exported={processed}/{effective_samples} "
                    f"time={elapsed:.2f}s"
                )

            exported_samples = max(exported_samples, processed)

    mapping_handle.close()
    print("finished")
    print(f"exported_samples={exported_samples} repeats={args.repeats} output_dir={output_dir}")


if __name__ == "__main__":
    main()


"""
python oa_reactdiff/evaluate/infer_30.py \
  --checkpoint oa_reactdiff/trainer/our_new_pretrained-ts1x-diff.ckpt \
  --dataset-path oa_reactdiff/data/data_new_split/test.pkl \
  --output-dir output/t1x_test_rollouts \
  --repeats 30 \
  --timesteps 250 \
  --resamplings 5 \
  --jump-length 5 \
  --batch-size 32
  
  
  
  
python oa_reactdiff/evaluate/infer_30.py \
  --checkpoint oa_reactdiff/trainer/our_new_pretrained-ts1x-rgd1-diff-h200.ckpt \
  --dataset-path oa_reactdiff/data/t1x_rgd1_mix/test.pkl \
  --output-dir output/mix_test_rollouts \
  --repeats 30 \
  --timesteps 250 \
  --resamplings 5 \
  --jump-length 5 \
  --batch-size 32
  
  
  
  
python oa_reactdiff/evaluate/infer_30.py \
  --checkpoint oa_reactdiff/trainer/our_new_pretrained-ts1x-rgd1-diff-h200-dim.ckpt \
  --dataset-path oa_reactdiff/data/t1x_rgd1_mix/test.pkl \
  --output-dir output/mix_dim_test_rollouts \
  --repeats 30 \
  --timesteps 250 \
  --resamplings 5 \
  --jump-length 5 \
  --batch-size 32



还加了批量运行脚本 run_gdb_raw_infer_30.sh。上传服务器后直接运行：
bash oa_reactdiff/evaluate/run_gdb_raw_infer_30.sh

默认使用：

oa_reactdiff/trainer/our_new_pretrained-ts1x-diff.ckpt

输出到：

output/gdb_raw_rollouts/GDB-10
output/gdb_raw_rollouts/GDB-17
  
"""
