import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING
from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa_reactdiff.trainer.pl_trainer import DDPMModule


DEFAULT_DATASET_PATH = PROJECT_ROOT / "oa_reactdiff" / "data" / "transition1x" / "valid_addprop.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xyz_from_ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export reactant/product xyz files and repeated TS predictions from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
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
    parser.add_argument("--resamplings", default=5, type=int, help="RePaint resamplings.")
    parser.add_argument("--jump-length", default=5, type=int, help="RePaint jump length.")
    parser.add_argument("--repeats", default=30, type=int, help="Number of TS predictions per sample.")
    parser.add_argument(
        "--single-frag-only",
        default=0,
        type=int,
        help="Whether to keep only single-fragment reactions (1 or 0). Default 0 keeps all.",
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
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


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
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    ddpm_trainer = DDPMModule(**checkpoint["hyper_parameters"])
    ddpm_trainer.load_state_dict(checkpoint["state_dict"])
    return ddpm_trainer.to(device)


def set_new_schedule_local(
    ddpm_trainer: DDPMModule,
    timesteps: int,
    device: torch.device,
    noise_schedule: str = "polynomial_2",
):
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
    ddpm_trainer: DDPMModule,
    resamplings: int,
    jump_length: int,
    frag_fixed=None,
):
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

    device = resolve_device(args.device)
    dataset_device = "cuda" if device.type == "cuda" else "cpu"
    single_frag_only = bool(args.single_frag_only)
    use_by_ind = bool(args.use_by_ind)
    save_true = bool(args.save_true)

    raw_dataset = load_pickle(dataset_path)
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
  
"""
