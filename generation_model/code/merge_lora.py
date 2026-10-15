"""Merge the AVC-Reward PEFT LoRA into a full LTX-2 checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def add_ltx_source() -> Path:
    script_dir = Path(__file__).resolve().parent
    root = Path(os.environ.get("LTX_V2_ROOT", script_dir / "ltx_v2")).expanduser().resolve()
    package_root = root / "packages" if (root / "packages").is_dir() else root
    for package in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
        source = package_root / package / "src"
        if not source.is_dir():
            raise FileNotFoundError(
                f"LTX runtime package is missing: {source}. Run generation_model/code/install_ltx_runtime.sh "
                "or set LTX_V2_ROOT to the pinned LTX-2 checkout."
            )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True, help="Matching full LTX-2 dev checkpoint (.safetensors)")
    parser.add_argument("--lora-dir", required=True, help="PEFT adapter directory with adapter_model.safetensors")
    parser.add_argument("--output-path", required=True, help="Path for the merged .safetensors checkpoint")
    parser.add_argument("--dtype", default="bf16", choices=["keep", "fp16", "bf16", "fp32"])
    parser.add_argument("--strict", action="store_true", help="Fail if any merged tensor is unmapped or mismatched")
    return parser.parse_args()


def load_full_safetensors(path: str):
    import safetensors

    state, metadata = {}, {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata()
        if source_metadata is not None:
            metadata = dict(source_metadata)
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state, metadata


def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file

    add_ltx_source()
    from ltx_trainer.model_loader import load_transformer

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    target_dtype = dtype_map.get(args.dtype)

    print(f"[1/4] Loading full checkpoint: {args.checkpoint_path}")
    full_state, metadata = load_full_safetensors(args.checkpoint_path)

    print("[2/4] Loading base transformer via LTX-2 model_loader ...")
    base_transformer = load_transformer(args.checkpoint_path, device="cpu", dtype=torch.float32)
    base_transformer.eval()

    print(f"[3/4] Loading LoRA from {args.lora_dir} and merging ...")
    peft_model = PeftModel.from_pretrained(base_transformer, args.lora_dir, is_trainable=False)
    lora_count = sum(1 for name, _ in peft_model.named_parameters() if "lora_" in name)
    print(f"    {lora_count} LoRA parameter tensors attached")
    if lora_count == 0:
        raise RuntimeError("No LoRA parameters attached. Adapter/base transformer keys may not match.")

    merged_model = peft_model.merge_and_unload()
    merged_state = merged_model.state_dict()
    prefix = "model.diffusion_model."
    replaced, not_found, shape_mismatch = 0, [], []

    for bare_key, tensor in merged_state.items():
        if not torch.is_tensor(tensor):
            continue
        target_key = prefix + bare_key
        if target_key not in full_state:
            not_found.append((bare_key, target_key))
            continue
        if full_state[target_key].shape != tensor.shape:
            shape_mismatch.append((bare_key, target_key, tuple(tensor.shape), tuple(full_state[target_key].shape)))
            continue
        output_tensor = tensor.detach().cpu().contiguous()
        output_tensor = output_tensor.to(target_dtype) if target_dtype is not None else output_tensor.to(full_state[target_key].dtype)
        full_state[target_key] = output_tensor
        replaced += 1

    print(f"    Replaced {replaced} tensors in full checkpoint")
    if not_found:
        print(f"    Unmapped tensors: {len(not_found)} (first 10):")
        for source_key, target_key in not_found[:10]:
            print(f"      {source_key} -> {target_key}")
    if shape_mismatch:
        print(f"    Shape mismatches: {len(shape_mismatch)} (first 10):")
        for source_key, target_key, source_shape, target_shape in shape_mismatch[:10]:
            print(f"      {source_key} -> {target_key}  {source_shape} vs {target_shape}")
    if args.strict and (not_found or shape_mismatch):
        raise RuntimeError("Strict mode: some LoRA tensors could not be mapped back.")
    if replaced == 0:
        raise RuntimeError("No checkpoint tensors were replaced; aborting.")

    print(f"[4/4] Saving to {args.output_path}")
    output_metadata = {str(key): str(value) for key, value in metadata.items()}
    output_metadata.update(
        {
            "lora_merged_fsdp": "true",
            "source_checkpoint": Path(args.checkpoint_path).name,
            "source_lora_dir": Path(args.lora_dir).name,
            "dtype_policy": args.dtype,
            "replaced_tensors": str(replaced),
        }
    )
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(full_state, str(output_path), metadata=output_metadata)
    print(f"[OK] {output_path} ({len(full_state)} tensors, {replaced} replaced)")


def main() -> None:
    args = parse_args()
    import torch

    with torch.no_grad():
        merge(args)


if __name__ == "__main__":
    main()
