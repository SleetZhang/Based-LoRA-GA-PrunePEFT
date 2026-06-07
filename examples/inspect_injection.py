#!/usr/bin/env python
"""
inspect_injection.py — READ-ONLY adapter-injection inspector.

What it does
------------
For each adapter config (lora / dora / bottleneck / combined) it builds the PEFT
model through THIS repo's own `controller.create_prunepeft_config` + `peft.get_peft_model`
— the exact path `controller.py` uses — then walks the model and prints:
  * which q/k/v/o linears actually became LoRA/DoRA layers (and on which layers)
  * which self_attn/mlp blocks got wrapped with a bottleneck adapter
  * the resolved `target_modules`, the wrapper class, and trainable-param count.

Because each repo has its own `create_prunepeft_config`, dropping this same file
into the modified repo vs. the original repo will faithfully show their difference
(e.g. combined LoRA on q/v/k/o vs. only q/v).

What it does NOT do
-------------------
  * It does NOT train, and does NOT read any running training process — separate
    Python processes have isolated memory. It independently *reproduces* the same
    deterministic injection in its own process.
  * It loads the model on CPU by default, so it will NOT touch a GPU that may be
    running a training in another terminal.
  * It shows the BUILD-TIME (pre-pruning) structure. For the combined config this
    is the warmup view (all 32 layers). The post-pruning layer set is data/seed
    dependent and only produced by a real `controller.sh` run.

Usage (run from the repo root so `examples/` is on sys.path[0]):
    python examples/inspect_injection.py
    python examples/inspect_injection.py --configs lora,dora
    python examples/inspect_injection.py --device cuda     # optional; default cpu
"""
import argparse
import gc
import re
from collections import defaultdict

import torch

# `examples/` is sys.path[0] when launched as `python examples/inspect_injection.py`,
# so these resolve to THIS repo's modules (the whole point of the comparison).
from controller import create_prunepeft_config
from utils import initialize_text_to_text_model
from peft import get_peft_model

MODEL_ID = "ckpts/pretrained/Llama-2-7b-hf"
NUM_LAYERS = 32
ALL_LAYERS = list(range(NUM_LAYERS))
TARGET_MODULES = "q_proj,v_proj,k_proj,o_proj"

# Common knobs, matching the *.sh wrappers.
COMMON = dict(
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bottleneck_size=32,
    bottleneck_dropout=0.1,
    init_bottleneck_weights=True,
    target_modules=TARGET_MODULES,
    bias="none",
)

# config key -> extra kwargs (mirrors controller intent at build time, pre-pruning).
CONFIGS = {
    "lora": dict(adapter_types=["lora"], lora_layers=ALL_LAYERS),
    "dora": dict(adapter_types=["dora"]),
    "bottleneck": dict(adapter_types=["bottleneck"], adapter_layers=ALL_LAYERS),
    "combined": dict(
        adapter_types=["lora", "bottleneck"],
        lora_layers=ALL_LAYERS,
        adapter_layers=ALL_LAYERS,
    ),
}

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _layer_idx(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _fmt(layer_set):
    xs = sorted(x for x in layer_set if x is not None)
    return f"{len(xs):>2d} layers {xs}" if xs else "none"


def inspect(model):
    """Walk the wrapped model and collect adapter placement."""
    lora_by_suffix = defaultdict(set)   # 'q_proj' -> {layer indices}
    dora = False
    block_by_kind = defaultdict(set)    # 'self_attn'/'mlp' -> {layer indices}
    bott_linear = defaultdict(set)      # bottleneck-as-linear, if any
    for name, module in model.named_modules():
        la = getattr(module, "lora_A", None)
        if la is not None and hasattr(la, "__len__") and len(la) > 0:
            lora_by_suffix[name.split(".")[-1]].add(_layer_idx(name))
            mv = getattr(module, "lora_magnitude_vector", None)
            if mv is not None and hasattr(mv, "__len__") and len(mv) > 0:
                dora = True
        cls = type(module).__name__
        if cls == "BlockWithAdapter":
            block_by_kind[name.split(".")[-1]].add(_layer_idx(name))
        elif cls == "BottleneckLinear":
            bott_linear[name.split(".")[-1]].add(_layer_idx(name))
    return lora_by_suffix, dora, block_by_kind, bott_linear


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one(key, device):
    kwargs = dict(COMMON)
    kwargs.update(CONFIGS[key])
    adapter_types = kwargs["adapter_types"]

    print("=" * 100)
    print(f"### CONFIG '{key}'  ->  --adapter_types \"{','.join(adapter_types)}\"")

    base_model, _tok = initialize_text_to_text_model(MODEL_ID, "CausalLM", "bf16")
    if device != "cpu":
        base_model = base_model.to(device)

    peft_config = create_prunepeft_config(model=base_model, **kwargs)
    model = get_peft_model(base_model, peft_config)

    cfg = model.peft_config.get("default", peft_config)
    print(f"wrapper                : {type(model).__name__}  /  base_model: {type(model.base_model).__name__}")
    print(f"resolved target_modules: {getattr(cfg, 'target_modules', None)}")

    lora_by_suffix, dora, block_by_kind, bott_linear = inspect(model)
    kind = "DoRA" if dora else "LoRA"

    if lora_by_suffix:
        print(f"{kind} linears:")
        ordered = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for suf in ordered + [s for s in lora_by_suffix if s not in ordered]:
            if suf in lora_by_suffix:
                print(f"    {suf:8s}: {_fmt(lora_by_suffix[suf])}")
    else:
        print(f"{kind} linears        : none")

    if block_by_kind:
        print("Bottleneck blocks:")
        for k in ("self_attn", "mlp"):
            if k in block_by_kind:
                print(f"    {k:10s}: {_fmt(block_by_kind[k])}")
    else:
        print("Bottleneck blocks      : none")

    if bott_linear:
        print("Bottleneck linears:")
        for suf, s in bott_linear.items():
            print(f"    {suf:8s}: {_fmt(s)}")

    tp = trainable_params(model)
    n_lora = sum(len(v) for v in lora_by_suffix.values())
    n_block = sum(len(v) for v in block_by_kind.values())
    print(f"trainable params       : {tp:,}")
    print()

    summary = dict(
        key=key,
        wrapper=type(model).__name__,
        kind=kind if lora_by_suffix else "-",
        n_lora=n_lora,
        n_block=n_block,
        trainable=tp,
        target_modules=getattr(cfg, "target_modules", None),
    )

    del model, base_model, peft_config
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    return summary


def main():
    global MODEL_ID
    ap = argparse.ArgumentParser(description="Read-only adapter-injection inspector.")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, won't touch GPU) or cuda")
    ap.add_argument("--model_id", default=MODEL_ID)
    args = ap.parse_args()

    MODEL_ID = args.model_id

    keys = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Choose from {list(CONFIGS)}")

    print(f"model_id = {MODEL_ID}   device = {args.device}   (CPU default = won't disturb a training GPU)")
    summaries = [run_one(k, args.device) for k in keys]

    print("=" * 100)
    print("SUMMARY")
    print(f"{'config':<10} {'wrapper':<22} {'adapter':<6} {'#lora':>6} {'#block':>7} {'trainable':>14}  target_modules")
    for s in summaries:
        print(f"{s['key']:<10} {s['wrapper']:<22} {s['kind']:<6} "
              f"{s['n_lora']:>6} {s['n_block']:>7} {s['trainable']:>14,}  {s['target_modules']}")


if __name__ == "__main__":
    main()
