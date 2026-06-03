from typing import List, Optional, Tuple
from uuid import uuid4
import argparse
import os
import shutil
import torch

from pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet


os.environ.setdefault("WANDB_MODE", "offline")


def parse_args():
    parser = argparse.ArgumentParser()

    def str2bool(value):
        value = value.lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"expected boolean value, got {value}")

    parser.add_argument("--datadir", type=str, default=None)
    parser.add_argument("--bz", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument(
        "--val_sampling_mode",
        choices=["first_batch", "all", "none"],
        default=None,
    )
    parser.add_argument("--single_frag_only", type=str2bool, default=None)
    parser.add_argument("--use_by_ind", type=str2bool, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--allow_existing_run_dir", type=str2bool, default=False)
    return parser.parse_args()


def resolve_trainer_runtime(args):
    cuda_devices = torch.cuda.device_count()
    launched_with_torchrun = "LOCAL_RANK" in os.environ
    num_nodes = args.num_nodes or int(os.environ.get("PET_NNODES", os.environ.get("NNODES", "1")))

    if cuda_devices > 0:
        accelerator = "gpu"
        devices = 1 if launched_with_torchrun else (args.devices or cuda_devices)
        if launched_with_torchrun or devices > 1 or num_nodes > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = None
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None

    return accelerator, devices, num_nodes, strategy, cuda_devices, launched_with_torchrun


args = parse_args()


model_type = "leftnet"
version = "0"
project = args.project or "OAReactDiff"

# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    lr=2.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 200
T_mult = 2
training_config = dict(
    datadir=args.datadir or "../../data/transition1x_rebuild/",
    remove_h=False,
    bz=64,
    num_workers=6,       #建议值不一定需要改
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=False,
    reflection=False,
    single_frag_only=False,
    only_ts=False,
    val_sampling_mode="all",  # "first_batch", "all", or "none"
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),  # step
)
training_data_frac = 1.0

node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["R", "TS", "P"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "TS1x"
enforce_same_encoding = None
scales = [1.0, 2.0, 1.0]
fixed_idx: Optional[List] = None
eval_epochs = 10
save_every = args.save_every or 10

if args.bz is not None:
    training_config["bz"] = args.bz
if args.num_workers is not None:
    training_config["num_workers"] = args.num_workers
if args.gradient_clip_val is not None:
    training_config["gradient_clip_val"] = args.gradient_clip_val
if args.single_frag_only is not None:
    training_config["single_frag_only"] = args.single_frag_only
if args.use_by_ind is not None:
    training_config["use_by_ind"] = args.use_by_ind
if args.val_sampling_mode is not None:
    training_config["val_sampling_mode"] = args.val_sampling_mode
if args.lr is not None:
    optimizer_config["lr"] = args.lr

# ----Normalizer---
norm_values: Tuple = (1.0, 1.0, 1.0)
norm_biases: Tuple = (0.0, 0.0, 0.0)

# ---Schedule---
noise_schedule: str = "cosine"
timesteps: int = 5000
precision: float = 1e-5

norms = "_".join([str(x) for x in norm_values])
run_name = args.run_name or f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

seed_everything(42, workers=True)
ddpm = DDPMModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    norm_values,
    norm_biases,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=None,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
run_dir = os.path.join("checkpoint", project, run_name)
ckpt_path = os.path.join(run_dir, "ckpts")
log_path = os.path.join(run_dir, "logs")
if os.path.exists(run_dir) and not args.allow_existing_run_dir:
    raise FileExistsError(
        f"Run directory already exists: {run_dir}. "
        "Use a new --run_name or pass --allow_existing_run_dir true to reuse it."
    )
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
trainer = None
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
        save_dir=log_path,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False)
    except:
        pass

earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)
full_val_sampling = training_config.get("val_sampling_mode") == "all"
if full_val_sampling:
    checkpoint_callback = ModelCheckpoint(
        monitor="val-rmsd-median",
        mode="min",
        dirpath=ckpt_path,
        filename="ddpm-{epoch:03d}-{val-rmsd-median:.4f}",
        every_n_epochs=save_every,
        save_top_k=1,
        save_last=True,
    )
else:
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="ddpm-{epoch:03d}",
        every_n_epochs=save_every,
        save_top_k=0,
        save_last=True,
    )
lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))

shutil.copy(f"../model/{model_type}.py", os.path.join(run_dir, f"{model_type}.py"))

print("config: ", config)

(
    accelerator,
    devices,
    num_nodes,
    strategy,
    cuda_devices,
    launched_with_torchrun,
) = resolve_trainer_runtime(args)

print(
    "trainer devices: "
    f"accelerator={accelerator}, devices={devices}, num_nodes={num_nodes}, "
    f"strategy={strategy.__class__.__name__ if strategy is not None else None}, "
    f"visible_cuda_devices={cuda_devices}, torchrun={launched_with_torchrun}"
)

trainer = Trainer(
    max_epochs=args.max_epochs,
    accelerator=accelerator,
    deterministic=False,
    devices=devices,
    num_nodes=num_nodes,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    logger=wandb_logger,
    accumulate_grad_batches=args.accumulate_grad_batches,
    gradient_clip_val=training_config["gradient_clip_val"],
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
# trainer.fit(ddpm, ckpt_path="<path-to-resume-checkpoint>")

if full_val_sampling and checkpoint_callback.best_model_path:
    final_ckpt_path = os.path.join(ckpt_path, "best.ckpt")
    shutil.copyfile(checkpoint_callback.best_model_path, final_ckpt_path)
    print(
        "Saved final checkpoint from best validation sampling median RMSD: "
        f"{checkpoint_callback.best_model_path} -> {final_ckpt_path}"
    )
