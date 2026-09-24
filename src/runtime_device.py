"""CPU-first device selection for reproducible experiment scripts."""
import os
import torch


def get_device():
    device = os.environ.get("PRISM_DEVICE", "cpu").lower()
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("PRISM_DEVICE must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use PRISM_DEVICE=cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use PRISM_DEVICE=cpu")
    return device
