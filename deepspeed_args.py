"""Batteries-included DeepSpeed + Accelerate setup.

The goal of this module is that distributed mixed-precision training "just
works" without hand-writing a DeepSpeed json config.  It is intentionally
opinionated: the defaults are tuned for maximum GPU-memory savings (ZeRO-3
with CPU offload of optimizer states and parameters, gradient checkpointing,
automatic mixed precision) while staying compatible with plain custom models
(not only HuggingFace transformers).

Typical use::

    from kemsekov_torch.deepspeed_args import deepspeed_args
    from kemsekov_torch.train import train

    args = deepspeed_args(model)          # sets up ZeRO-3 + AMP + checkpointing
    train(model, train_loader, test_loader, compute_loss, "runs/exp",
          accelerate_args=args, num_epochs=10)

For multi-GPU launch use a launcher (the arguments work as-is)::

    torchrun --nproc_per_node=2 script.py
    # or
    accelerate launch --num_processes 2 script.py
    # or, from a notebook, use accelerate.notebook_launcher

Notes
-----
* ``torch.compile`` is disabled by default (the returned ``dynamo_plugin``
  has ``backend="no"``) because DeepSpeed's partitioned parameters cannot be
  traced by Dynamo in this stack (empty placeholder weights under ZeRO-3 and
  ``None`` device synchronizations).  Pass ``use_compile=True`` to force the
  inductor plugin.
* The default optimizer is ``Muon`` (MuonWithAuxAdam), the current SOTA
  optimizer natively supported by DeepSpeed >= 0.19.  ``reduce_scatter`` is
  disabled automatically because DeepSpeed forbids combining it with Muon.
  On older DeepSpeed builds the helper falls back to ``OneBitAdam`` when
  available and finally to ``AdamW`` (which becomes DeepSpeedCPUAdam when
  optimizer offload is on).
* If bf16 is not supported by the current GPU (e.g. Volta), fp16 is used
  instead automatically.
"""

import os
from typing import Any, Dict, Literal, Optional

__all__ = ["deepspeed_args", "enable_gradient_checkpointing"]


def _ensure_distributed_env() -> None:
    """Pretend we are a one-process distributed launch.

    DeepSpeed falls back to MPI discovery when ``RANK``/``WORLD_SIZE``/...
    are missing.  On machines where OpenMPI is only partially installed
    (no ``orted``/``mpirun``) that discovery aborts the whole process with
    ``MPI_ERRORS_ARE_FATAL`` -- in a Jupyter notebook this looks like a
    mysterious kernel crash.  Real launchers (torchrun/accelerate/deepspeed)
    set these variables themselves, so we only fill the gaps.
    """
    if "MASTER_PORT" not in os.environ:
        # Pick a free port instead of hardcoding one: a previously interrupted
        # notebook_launcher can leave 29500 occupied inside the same kernel.
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _resolve_mixed_precision(mixed_precision: str) -> str:
    mixed_precision = mixed_precision.lower()
    if mixed_precision in ("no", "fp32", "float32"):
        return "no"
    if mixed_precision not in ("bf16", "fp16"):
        raise ValueError(
            f"mixed_precision must be one of 'bf16', 'fp16', 'no', got {mixed_precision!r}"
        )
    if mixed_precision == "bf16":
        try:
            import torch

            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                print(
                    "[deepspeed_args] bf16 is not supported by this GPU, "
                    "falling back to fp16 mixed precision."
                )
                return "fp16"
        except Exception:
            return "fp16"
    return mixed_precision


def _resolve_optimizer(optimizer: Optional[str]) -> str:
    """Pick the best optimizer the installed DeepSpeed ships.

    Preference order: Muon (current SOTA, native in DeepSpeed >= 0.19) ->
    OneBitAdam (old DeepSpeed, needs MPI) -> AdamW (always available, and
    becomes DeepSpeedCPUAdam when optimizer offload is on).
    """
    if optimizer is not None:
        return optimizer
    try:
        from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS

        if "muon" in [name.lower() for name in DEEPSPEED_OPTIMIZERS]:
            return "Muon"
    except Exception:
        pass
    try:
        from deepspeed.ops.adam import OneBitAdam  # noqa: F401

        return "OneBitAdam"
    except Exception:
        return "AdamW"


_CHECKPOINT_CONTAINER_NAMES = (
    "middle",
    "layers",
    "blocks",
    "h",
    "transformer.h",
    "model.layers",
    "model.decoder.layers",
)


def _find_checkpoint_container(model, container_path="auto"):
    import torch.nn as nn

    if container_path != "auto":
        container = model
        for part in container_path.split("."):
            container = getattr(container, part, None)
            if container is None:
                return None
        return container
    for name in _CHECKPOINT_CONTAINER_NAMES:
        container = model
        for part in name.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if container is not None and isinstance(container, (nn.Sequential, nn.ModuleList)):
            return container
    return None


def enable_gradient_checkpointing(model, container_path="auto") -> bool:
    """Turn on activation/gradient checkpointing on a model.

    Three mechanisms, tried in order:

    1. HF-style ``model.gradient_checkpointing_enable()``.
    2. A ``model.gradient_checkpointing`` boolean attribute.
    3. Generic fallback that needs no model changes: find the layer container
       (``middle``, ``layers``, ``transformer.h``, ... or ``container_path``)
       and patch its ``forward`` so every child runs under
       ``torch.utils.checkpoint``.  This keeps custom models untouched while
       still trading compute for a large drop in activation memory.

    Returns ``True`` when checkpointing was enabled.
    """
    import types

    import torch
    from torch.utils.checkpoint import checkpoint

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return True
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
        return True

    container = _find_checkpoint_container(model, container_path)
    if container is None:
        print(
            "[deepspeed_args] WARNING: could not find a layer container to "
            "checkpoint (tried "
            + ", ".join(_CHECKPOINT_CONTAINER_NAMES)
            + "); ignoring the request."
        )
        return False

    if getattr(container, "_deepspeed_args_checkpointed", False):
        return True

    def checkpointed_forward(self, x):
        if not self.training:
            for layer in self:
                x = layer(x)
            return x
        for layer in self:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

    container.forward = types.MethodType(checkpointed_forward, container)
    container._deepspeed_args_checkpointed = True
    print(
        "[deepspeed_args] gradient checkpointing enabled on "
        f"{type(model).__name__}.{type(container).__name__} layers."
    )
    return True


def deepspeed_args(
    model=None,
    *,
    # precision
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16",
    # ZeRO / memory
    zero_stage: int = 3,
    offload_optimizer: bool = True,
    offload_param: bool = True,
    offload_optimizer_device: Literal["cpu", "nvme", "none"] = "cpu",
    offload_param_device: Literal["cpu", "nvme", "none"] = "cpu",
    pin_memory: bool = True,
    gradient_checkpointing: bool = False,
    checkpoint_container: str = "auto",
    activation_checkpointing: bool = False,
    # optimization
    optimizer: Optional[str] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: Optional[float] = 1.0,
    # batching
    split_batches: bool = False,
    # compilation
    use_compile: Optional[bool] = None,
    compile_backend: str = "inductor",
    compile_mode: str = "default",
    compile_fullgraph: bool = False,
    compile_dynamic: bool = True,
    # misc
    zero3_init_flag: bool = True,
    overlap_comm: bool = True,
    contiguous_gradients: bool = True,
    reduce_scatter: bool = True,
    steps_per_print: int = 1000000,
    extra_deepspeed_config: Optional[Dict[str, Any]] = None,
    extra_accelerate_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ready-to-use ``accelerate`` arguments for DeepSpeed training.

    Parameters
    ----------
    model:
        Optional model.  When given and ``gradient_checkpointing=True`` the
        model is switched into checkpointing mode in-place (this is what
        actually saves activation memory -- DeepSpeed's own
        ``activation_checkpointing`` config only applies to pipeline/HF
        models).  HF models use their own API; for custom models the layer
        container (``middle``/``layers``/``transformer.h``/..., or
        ``checkpoint_container``) is patched to checkpoint every child layer,
        so the model code itself stays untouched.
    mixed_precision:
        ``"bf16"`` (default), ``"fp16"`` or ``"no"``.  Automatically falls
        back to fp16 when bf16 is unsupported by the GPU.
    zero_stage:
        ZeRO optimization stage (``3`` by default, which shards parameters,
        gradients and optimizer state across all GPUs).
    offload_optimizer / offload_param:
        Offload optimizer state / ZeRO-3 parameter shards to CPU (or NVMe).
        Enabled by default for maximum memory savings.
    gradient_checkpointing:
        Enable activation checkpointing (see ``model``).  Trades ~30% compute
        for a large drop in activation memory; required for long sequences /
        big batches.
    checkpoint_container:
        Attribute path of the layer container to patch when the model has no
        native checkpointing API (default ``"auto"``).
    activation_checkpointing:
        Put DeepSpeed's ``activation_checkpointing`` section into the config
        (useful for HF/pipeline models, harmless no-op otherwise).
    optimizer:
        DeepSpeed optimizer name.  ``None`` (default) picks ``Muon`` when the
        installed DeepSpeed supports it (native since 0.19), then
        ``OneBitAdam``, then ``AdamW``.  Pass e.g. ``optimizer="AdamW"`` to
        opt out of Muon.
    use_compile:
        Whether to return a torch.compile (TorchDynamo) plugin.  ``None``
        (default) disables it: tracing DeepSpeed's partitioned parameters
        fails in the current stack, and an uncompiled model is always
        correct.  Set to ``True`` to force the inductor plugin.
    split_batches:
        Split each dataloader batch across processes (halves per-GPU
        activation memory at the cost of a larger effective batch otherwise).
    zero3_init_flag:
        Let accelerate install a ``deepspeed.zero.Init`` weakref (only
        meaningful for transformers models, harmless otherwise).
    extra_deepspeed_config:
        Dict merged on top of the generated DeepSpeed config.
    extra_accelerate_args:
        Dict merged on top of the returned accelerate arguments.

    Returns
    -------
    dict
        Keyword arguments for ``accelerate.Accelerator(**args)`` (and for
        ``kemsekov_torch.train.train(..., accelerate_args=args)``).
    """
    from accelerate import DeepSpeedPlugin
    from accelerate.utils import TorchDynamoPlugin

    _ensure_distributed_env()

    mixed_precision = _resolve_mixed_precision(mixed_precision)
    optimizer_name = _resolve_optimizer(optimizer)
    if optimizer_name.lower() == "muon" and reduce_scatter:
        # DeepSpeed raises "Muon and reduce scatter cannot be used together".
        print("[deepspeed_args] Muon requires reduce_scatter=False, disabling reduce_scatter.")
        reduce_scatter = False

    if model is not None and gradient_checkpointing:
        enable_gradient_checkpointing(model, container_path=checkpoint_container)

    offload_optimizer_device = None if not offload_optimizer else offload_optimizer_device
    offload_param_device = None if not offload_param else offload_param_device

    ds_config: Dict[str, Any] = {
        # accelerate fills the "auto" values from the dataloaders
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": steps_per_print,
        "optimizer": {
            "type": optimizer_name,
            "params": {"lr": learning_rate, "weight_decay": weight_decay},
        },
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": overlap_comm,
            "contiguous_gradients": contiguous_gradients,
            "reduce_scatter": reduce_scatter,
        },
    }
    if gradient_clipping is not None:
        ds_config["gradient_clipping"] = float(gradient_clipping)
    if offload_optimizer_device is not None:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": offload_optimizer_device,
            "pin_memory": pin_memory,
        }
    if offload_param_device is not None:
        ds_config["zero_optimization"]["offload_param"] = {
            "device": offload_param_device,
            "pin_memory": pin_memory,
        }
    if activation_checkpointing:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
        }
    if extra_deepspeed_config:
        for key, value in extra_deepspeed_config.items():
            if isinstance(value, dict) and isinstance(ds_config.get(key), dict):
                ds_config[key].update(value)
            else:
                ds_config[key] = value

    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config,
        zero_stage=zero_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        zero3_init_flag=zero3_init_flag,
        offload_optimizer_device=offload_optimizer_device,
        offload_param_device=offload_param_device,
    )

    if use_compile is None:
        use_compile = False
        print(
            "[deepspeed_args] torch.compile is disabled by default because "
            "DeepSpeed's partitioned parameters are not traceable by Dynamo "
            "in this stack. Pass use_compile=True to force it."
        )

    if use_compile:
        dynamo_plugin = TorchDynamoPlugin(
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
    else:
        dynamo_plugin = TorchDynamoPlugin(backend="no")

    accelerate_args: Dict[str, Any] = {
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "split_batches": split_batches,
        "deepspeed_plugin": deepspeed_plugin,
        "dynamo_plugin": dynamo_plugin,
    }
    if extra_accelerate_args:
        accelerate_args.update(extra_accelerate_args)

    return accelerate_args
