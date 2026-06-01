import argparse
import pickle
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.evaluate.infer_30 import (
    inpaint_batch_local,
    load_ddpm_from_checkpoint,
    safe_value,
    set_new_schedule_local,
    split_by_sample,
    write_xyz_block,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "oa_reactdiff" / "trainer" / "our_new_pretrained-ts1x-diff.ckpt"
)
DEFAULT_TAR_PATHS = [
    PROJECT_ROOT / "oa_reactdiff" / "data" / "GDB-10-rxn_raw.tar.gz",
    PROJECT_ROOT / "oa_reactdiff" / "data" / "GDB-17-rxn_raw.tar.gz",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "gdb_raw_rollouts"

ATOM_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}
FRAGMENTS = {
    "reactant": "R.xyz",
    "transition_state": "TS.xyz",
    "product": "P.xyz",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert GDB raw reaction tarballs to ProcessedTS1x-compatible pkl "
            "files and export repeated TS inpainting predictions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        type=str,
        help="Path to the trained .ckpt file.",
    )
    parser.add_argument(
        "--tar-paths",
        nargs="+",
        default=[str(path) for path in DEFAULT_TAR_PATHS],
        help="One or more raw reaction .tar.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        type=str,
        help="Directory for converted pkl caches and exported xyz files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help='Device to use: "auto", "cuda", or "cpu".',
    )
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size.")
    parser.add_argument("--timesteps", default=250, type=int, help="Diffusion timesteps.")
    parser.add_argument("--resamplings", default=5, type=int, help="RePaint resamplings.")
    parser.add_argument("--jump-length", default=5, type=int, help="RePaint jump length.")
    parser.add_argument("--repeats", default=30, type=int, help="Predictions per sample.")
    parser.add_argument(
        "--single-frag-only",
        default=0,
        type=int,
        help=(
            "Whether to keep only single-fragment reactions according to the "
            "converted metadata. Default 0 keeps every raw reaction."
        ),
    )
    parser.add_argument(
        "--use-by-ind",
        default=1,
        type=int,
        help="Whether to filter by the converted use_ind split. Default includes all rows.",
    )
    parser.add_argument(
        "--max-samples",
        default=-1,
        type=int,
        help="Limit samples per tarball. Use -1 for all samples.",
    )
    parser.add_argument("--num-workers", default=0, type=int, help="Dataloader workers.")
    parser.add_argument(
        "--force-rebuild",
        default=0,
        type=int,
        help="Rebuild converted pkl caches even when they already exist.",
    )
    parser.add_argument(
        "--prepare-only",
        default=0,
        type=int,
        help="Only build converted pkl caches; do not load the model or infer.",
    )
    parser.add_argument(
        "--save-true",
        default=1,
        type=int,
        help="Also save raw R/TS/P into true_rts_p.xyz.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def dataset_name_from_tar(tar_path: Path) -> str:
    name = tar_path.name
    for suffix in [".tar.gz", ".tgz", ".tar"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return tar_path.stem


def parse_xyz_text(text: str, member_name: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty xyz file: {member_name}")

    natoms = int(lines[0])
    atom_lines = lines[1:]
    if atom_lines and not atom_lines[0].split()[0] in ATOM_NUMBERS:
        atom_lines = atom_lines[1:]
    if len(atom_lines) < natoms:
        raise ValueError(f"Expected {natoms} atoms but found {len(atom_lines)} in {member_name}")

    charges = []
    positions = []
    for line in atom_lines[:natoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed xyz atom line in {member_name}: {line}")
        symbol = parts[0]
        if symbol not in ATOM_NUMBERS:
            raise ValueError(f"Unsupported element {symbol!r} in {member_name}")
        charges.append(np.int32(ATOM_NUMBERS[symbol]))
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return charges, np.asarray(positions, dtype=np.float32)


def molecular_formula(charges):
    counts = {symbol: 0 for symbol in ATOM_NUMBERS}
    number_to_symbol = {number: symbol for symbol, number in ATOM_NUMBERS.items()}
    for charge in charges:
        counts[number_to_symbol[int(charge)]] += 1

    pieces = []
    for symbol in ["C", "H", "N", "O", "F"]:
        count = counts[symbol]
        if count:
            pieces.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(pieces)


def empty_fragment_record():
    return {
        "num_atoms": [],
        "charges": [],
        "fragments": [],
        "positions": [],
        "rxn": [],
        "wB97x_6-31G(d).energy": [],
        "wB97x_6-31G(d).atomization_energy": [],
        "wB97x_6-31G(d).forces": [],
        "formula": [],
    }


def find_reactions(tar: tarfile.TarFile):
    grouped = defaultdict(dict)
    for member in tar.getmembers():
        if not member.isfile():
            continue
        path = Path(member.name)
        if path.name in {"R.xyz", "TS.xyz", "P.xyz"}:
            grouped[str(path.parent)][path.name] = member

    complete = []
    required = set(FRAGMENTS.values())
    for reaction_dir, members in grouped.items():
        if required.issubset(members):
            complete.append((reaction_dir, members))
    complete.sort(key=lambda item: natural_key(item[0]))
    return complete


def convert_raw_tar_to_pkl(tar_path: Path, pkl_path: Path, force_rebuild: bool = False) -> Path:
    if pkl_path.is_file() and not force_rebuild:
        print(f"using cached converted dataset: {pkl_path}")
        return pkl_path

    print(f"converting raw tar: {tar_path}")
    dataset = {
        "reactant": empty_fragment_record(),
        "transition_state": empty_fragment_record(),
        "product": empty_fragment_record(),
        "single_fragment": [],
        "use_ind": [],
    }

    with tarfile.open(tar_path, "r:gz") as tar:
        reactions = find_reactions(tar)
        for row_idx, (reaction_dir, members) in enumerate(reactions):
            rxn_id = Path(reaction_dir).name
            reference_charges = None
            for fragment_name, xyz_name in FRAGMENTS.items():
                member = members[xyz_name]
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError(f"Cannot read {member.name}")
                text = handle.read().decode("utf-8", errors="replace")
                charges, positions = parse_xyz_text(text, member.name)
                if reference_charges is None:
                    reference_charges = charges
                elif list(charges) != list(reference_charges):
                    raise ValueError(
                        f"Atom order/charges differ across R/TS/P for {reaction_dir}"
                    )

                record = dataset[fragment_name]
                record["num_atoms"].append(len(charges))
                record["charges"].append(charges)
                record["fragments"].append([list(range(len(charges)))])
                record["positions"].append(positions)
                record["rxn"].append(rxn_id)
                record["wB97x_6-31G(d).energy"].append(np.nan)
                record["wB97x_6-31G(d).atomization_energy"].append(np.nan)
                record["wB97x_6-31G(d).forces"].append(np.zeros_like(positions))
                record["formula"].append(molecular_formula(charges))

            dataset["single_fragment"].append(1)
            dataset["use_ind"].append(np.int64(row_idx))

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as handle:
        pickle.dump(dataset, handle)
    print(f"wrote converted dataset: {pkl_path} ({len(dataset['use_ind'])} samples)")
    return pkl_path


def write_mapping_header(path: Path):
    handle = open(path, "w")
    handle.write("sample_id\trxn\n")
    return handle


def build_dataset(dataset_path: Path, device: torch.device, single_frag_only: bool, use_by_ind: bool):
    dataset_device = "cuda" if device.type == "cuda" else "cpu"
    return ProcessedTS1x(
        npz_path=str(dataset_path),
        center=True,
        pad_fragments=0,
        device=dataset_device,
        zero_charge=False,
        remove_h=False,
        single_frag_only=single_frag_only,
        swapping_react_prod=False,
        use_by_ind=use_by_ind,
        position_key="positions",
    )


def infer_dataset(
    dataset_path: Path,
    output_dir: Path,
    ddpm_trainer,
    device: torch.device,
    args,
):
    single_frag_only = bool(args.single_frag_only)
    use_by_ind = bool(args.use_by_ind)
    save_true = bool(args.save_true)

    dataset = build_dataset(dataset_path, device, single_frag_only, use_by_ind)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    effective_samples = len(dataset)
    if args.max_samples > 0:
        effective_samples = min(effective_samples, args.max_samples)
    print(
        f"dataset={dataset_path.name} samples={len(dataset)} "
        f"effective_samples={effective_samples} output_dir={output_dir}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_handle = write_mapping_header(output_dir / "sample_to_rxn_mapping.txt")

    exported_samples = 0
    with torch.no_grad():
        for repeat_idx in range(args.repeats):
            print(f"[{dataset_path.name}] repeat {repeat_idx + 1}/{args.repeats} starting")
            processed = 0
            sample_offset = 0

            for batch_idx, batch in enumerate(loader):
                if processed >= effective_samples:
                    break

                batch_start = time.time()
                out_samples, xh_fixed, fragments_nodes = inpaint_batch_local(
                    batch=batch,
                    ddpm_trainer=ddpm_trainer,
                    resamplings=args.resamplings,
                    jump_length=args.jump_length,
                    frag_fixed=[0, 2],
                )

                split_fixed = split_by_sample(xh_fixed, fragments_nodes)
                split_output = split_by_sample(out_samples, fragments_nodes)
                batch_size = len(split_output[1])
                batch_limit = min(batch_size, effective_samples - processed)

                for local_idx in range(batch_limit):
                    dataset_index = sample_offset + local_idx
                    rxn_id = safe_value(dataset.raw_dataset["reactant"]["rxn"][dataset_index])
                    sample_dir_name = f"sample_{dataset_index:05d}_{rxn_id}"
                    sample_dir = output_dir / sample_dir_name
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    if repeat_idx == 0:
                        mapping_handle.write(f"{sample_dir_name}\t{rxn_id}\n")
                        if save_true:
                            with open(sample_dir / "true_rts_p.xyz", "w") as handle:
                                write_xyz_block(handle, split_fixed[0][local_idx])
                                write_xyz_block(handle, split_fixed[1][local_idx])
                                write_xyz_block(handle, split_fixed[2][local_idx])

                    with open(sample_dir / f"rts_p_repeat_{repeat_idx:02d}.xyz", "w") as handle:
                        write_xyz_block(handle, split_fixed[0][local_idx])
                        write_xyz_block(handle, split_output[1][local_idx])
                        write_xyz_block(handle, split_fixed[2][local_idx])

                processed += batch_limit
                sample_offset += batch_size
                elapsed = time.time() - batch_start
                print(
                    f"[{dataset_path.name}] repeat={repeat_idx + 1}/{args.repeats} "
                    f"batch={batch_idx} exported={processed}/{effective_samples} "
                    f"time={elapsed:.2f}s"
                )

            exported_samples = max(exported_samples, processed)

    mapping_handle.close()
    print(f"finished {dataset_path.name}: exported_samples={exported_samples}")


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    converted_paths = []
    for tar_arg in args.tar_paths:
        tar_path = Path(tar_arg).resolve()
        dataset_name = dataset_name_from_tar(tar_path)
        converted_path = output_root / dataset_name / f"{dataset_name}.pkl"
        converted_paths.append(
            convert_raw_tar_to_pkl(
                tar_path=tar_path,
                pkl_path=converted_path,
                force_rebuild=bool(args.force_rebuild),
            )
        )

    if bool(args.prepare_only):
        print("prepare-only requested; skipping inference")
        return

    device = resolve_device(args.device)
    ddpm_trainer = load_ddpm_from_checkpoint(
        checkpoint_path=Path(args.checkpoint).resolve(),
        device=device,
    )
    ddpm_trainer = set_new_schedule_local(
        ddpm_trainer=ddpm_trainer,
        timesteps=args.timesteps,
        device=device,
    )
    ddpm_trainer.eval()

    for dataset_path in converted_paths:
        infer_dataset(
            dataset_path=dataset_path,
            output_dir=dataset_path.parent / "xyz",
            ddpm_trainer=ddpm_trainer,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()


"""
Example:

python oa_reactdiff/evaluate/infer_raw_gdb_30.py \
  --checkpoint oa_reactdiff/trainer/our_new_pretrained-ts1x-diff.ckpt \
  --tar-paths oa_reactdiff/data/GDB-10-rxn_raw.tar.gz oa_reactdiff/data/GDB-17-rxn_raw.tar.gz \
  --output-dir output/gdb_raw_rollouts \
  --repeats 30 \
  --timesteps 250 \
  --resamplings 5 \
  --jump-length 5 \
  --batch-size 32
"""
