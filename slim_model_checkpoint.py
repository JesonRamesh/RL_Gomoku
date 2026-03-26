"""
Create a slim inference-only .pt file from a training checkpoint.

Supports common formats:
- DQN full checkpoint with q/target/optimizer state
- AlphaZero-style wrapped checkpoint with `state_dict`
- Already-slim state_dict files

Examples:
  python slim_model_checkpoint.py \
      --input Model/rohan_model_160_epochs.pt \
      --output Model/rohan_model_160_epochs_slim.pt

  python slim_model_checkpoint.py \
      --input Model/rohan_model_160_epochs.pt \
      --output Model/rohan_model_160_epochs_slim_fp16.pt \
      --fp16
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch


DQN_WEIGHTS_KEY = "q_network_state_dict"
WRAPPED_STATE_DICT_KEY = "state_dict"


def _to_fp16_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            converted[key] = value.half()
        else:
            converted[key] = value
    return converted


def extract_inference_state_dict(checkpoint: Any) -> Dict[str, Any]:
    """Return a plain state_dict usable for inference-only loading."""
    if isinstance(checkpoint, dict):
        if DQN_WEIGHTS_KEY in checkpoint:
            # Full DQN training checkpoint -> save only online network weights.
            return checkpoint[DQN_WEIGHTS_KEY]

        if WRAPPED_STATE_DICT_KEY in checkpoint and isinstance(
            checkpoint[WRAPPED_STATE_DICT_KEY], dict
        ):
            # Wrapped checkpoint format.
            return checkpoint[WRAPPED_STATE_DICT_KEY]

        # Already state_dict-like (param name -> tensor).
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint

    raise ValueError(
        "Unsupported checkpoint format. Expected DQN full checkpoint, wrapped "
        "state_dict checkpoint, or plain state_dict."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a training checkpoint to slim inference-only .pt"
    )
    parser.add_argument("--input", required=True, help="Path to source .pt file")
    parser.add_argument(
        "--output", required=True, help="Path to output slim .pt file"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store floating-point weights in float16 for smaller files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = extract_inference_state_dict(checkpoint)

    if args.fp16:
        state_dict = _to_fp16_state_dict(state_dict)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(state_dict, args.output)

    in_size_mb = os.path.getsize(args.input) / (1024 * 1024)
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    pct = (1 - (out_size_mb / in_size_mb)) * 100 if in_size_mb > 0 else 0.0

    print(f"Input : {args.input} ({in_size_mb:.2f} MB)")
    print(f"Output: {args.output} ({out_size_mb:.2f} MB)")
    print(f"Reduced by: {pct:.1f}%")


if __name__ == "__main__":
    main()
