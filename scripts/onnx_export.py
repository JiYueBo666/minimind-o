"""
Export MiniMindOmni Thinker and Talker to ONNX with KV-cache support.

Usage:
    python scripts/onnx_export.py --load_from ./minimind-3o
    python scripts/onnx_export.py --load_from model --checkpoint ./out/sft_omni_768.pth
    python scripts/onnx_export.py --load_from ./minimind-3o --precision fp32
"""

import argparse
import os
import sys
import warnings

import torch
from torch import nn

__package__ = "scripts"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model_omni import MiniMindOmni, OmniConfig
from model.model_minimind import precompute_freqs_cis

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ONNX wrapper classes — unrolled for clean tracing
# ---------------------------------------------------------------------------

class ThinkerONNX(nn.Module):
    """Thinker: 8 layers with KV-cache as 16 individual (1, past, 4, 96) tensors."""

    def __init__(self, thinker, bridge_layer):
        super().__init__()
        self.L = thinker.layers
        self.norm = thinker.norm
        self.bridge_layer = bridge_layer

    def forward(
        self,
        h, cos, sin,
        pk0, pv0, pk1, pv1, pk2, pv2, pk3, pv3,
        pk4, pv4, pk5, pv5, pk6, pv6, pk7, pv7,
    ):
        pos = (cos, sin)
        h, (nk0, nv0) = self.L[0](h, pos, past_key_value=(pk0, pv0), use_cache=True)
        h, (nk1, nv1) = self.L[1](h, pos, past_key_value=(pk1, pv1), use_cache=True)
        h, (nk2, nv2) = self.L[2](h, pos, past_key_value=(pk2, pv2), use_cache=True)
        bridge = h
        h, (nk3, nv3) = self.L[3](h, pos, past_key_value=(pk3, pv3), use_cache=True)
        h, (nk4, nv4) = self.L[4](h, pos, past_key_value=(pk4, pv4), use_cache=True)
        h, (nk5, nv5) = self.L[5](h, pos, past_key_value=(pk5, pv5), use_cache=True)
        h, (nk6, nv6) = self.L[6](h, pos, past_key_value=(pk6, pv6), use_cache=True)
        h, (nk7, nv7) = self.L[7](h, pos, past_key_value=(pk7, pv7), use_cache=True)
        h = self.norm(h)
        return h, bridge, nk0, nv0, nk1, nv1, nk2, nv2, nk3, nv3, nk4, nv4, nk5, nv5, nk6, nv6, nk7, nv7


class TalkerONNX(nn.Module):
    """Talker: 4 layers with KV-cache as 8 individual tensors."""

    def __init__(self, talker):
        super().__init__()
        self.L = talker.layers
        self.norm = talker.norm

    def forward(self, h, cos, sin, pk0, pv0, pk1, pv1, pk2, pv2, pk3, pv3):
        pos = (cos, sin)
        h, (nk0, nv0) = self.L[0](h, pos, past_key_value=(pk0, pv0), use_cache=True)
        h, (nk1, nv1) = self.L[1](h, pos, past_key_value=(pk1, pv1), use_cache=True)
        h, (nk2, nv2) = self.L[2](h, pos, past_key_value=(pk2, pv2), use_cache=True)
        h, (nk3, nv3) = self.L[3](h, pos, past_key_value=(pk3, pv3), use_cache=True)
        h = self.norm(h)
        return h, nk0, nv0, nk1, nv1, nk2, nv2, nk3, nv3


# ---------------------------------------------------------------------------
# I/O names & dynamic axes
# ---------------------------------------------------------------------------

KV_HEAD_INFO = {"heads": 4, "dim": 96}  # num_key_value_heads, head_dim


def names_and_axes(n_layers, has_bridge=False):
    in_names = ["hidden_states", "freqs_cos", "freqs_sin"]
    out_names = ["h_out"]
    if has_bridge:
        out_names.append("bridge_states")
    for i in range(n_layers):
        in_names += [f"past_key_{i}", f"past_value_{i}"]
        out_names += [f"present_key_{i}", f"present_value_{i}"]

    axes = {
        "hidden_states": {1: "seq_len"},
        "freqs_cos": {0: "seq_len"},
        "freqs_sin": {0: "seq_len"},
    }
    for i in range(n_layers):
        axes[f"past_key_{i}"] = {1: "past_len"}
        axes[f"past_value_{i}"] = {1: "past_len"}
    for name in out_names:
        if name not in axes:
            axes[name] = {1: "seq_len"} if name == "h_out" or name == "bridge_states" else {1: "total_len"}
    return in_names, out_names, axes


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _disable_flash_attn(module):
    for m in module.modules():
        if hasattr(m, "flash"):
            m.flash = False


def _ensure_rope_buffers(model, device, dtype):
    for name, part in [("thinker", model.thinker), ("talker", model.talker)]:
        if name == "thinker":
            cfg = model.config
        else:
            cfg = model.talker.talker_config
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=cfg.head_dim, end=cfg.max_position_embeddings,
            rope_base=cfg.rope_theta, rope_scaling=cfg.rope_scaling,
        )
        part.freqs_cos.copy_(freqs_cos.to(dtype=dtype, device=device))
        part.freqs_sin.copy_(freqs_sin.to(dtype=dtype, device=device))


def _empty_kv(device, dtype, n_heads=4, head_dim=96):
    return torch.zeros(1, 0, n_heads, head_dim, dtype=dtype, device=device)


def _export(wrapper, args, in_names, out_names, dynamic_axes, path):
    torch.onnx.export(
        wrapper, args, path,
        input_names=in_names, output_names=out_names,
        dynamic_axes=dynamic_axes,
        opset_version=18, do_constant_folding=True, export_params=True,
        dynamo=False,  # Use legacy TorchScript tracer (Dynamo optimizer has string-tensor bug)
    )
    import onnx
    onnx.checker.check_model(path)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"  -> {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    if args.load_from == "model":
        config = OmniConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            flash_attn=False,
        )
        model = MiniMindOmni(config, audio_encoder_path=None, vision_model_path=None)
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state, strict=False)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True,
            torch_dtype=torch.float16 if args.precision == "fp16" else torch.float32,
        )
    model = model.eval()
    if args.precision == "fp16":
        model = model.half()
    return model.to(args.device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export MiniMindOmni to ONNX")
    parser.add_argument("--load_from", default="./minimind-3o")
    parser.add_argument("--checkpoint", default="./out/sft_omni_768.pth")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--output_dir", default="./scripts")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model ({args.precision}, device={args.device}) ...")
    model = load_model(args)
    dtype = next(model.thinker.layers[0].parameters()).dtype
    device = args.device
    n_kv = KV_HEAD_INFO["heads"]
    hd = KV_HEAD_INFO["dim"]

    _disable_flash_attn(model.thinker)
    _disable_flash_attn(model.talker)
    _ensure_rope_buffers(model, device, dtype)

    seq_len = 4
    ekv = _empty_kv(device, dtype, n_kv, hd)

    # --- Thinker ---
    t_wrapper = ThinkerONNX(model.thinker, model.config.bridge_layer).to(device)
    t_hidden = torch.randn(1, seq_len, 768, dtype=dtype, device=device)
    t_cos = model.thinker.freqs_cos[:seq_len].to(dtype=dtype, device=device)
    t_sin = model.thinker.freqs_sin[:seq_len].to(dtype=dtype, device=device)
    t_args = (t_hidden, t_cos, t_sin, *[ekv] * 16)
    t_ins, t_outs, t_axes = names_and_axes(8, has_bridge=True)
    print(f"Exporting Thinker ({len(model.thinker.layers)} layers) ...")
    _export(t_wrapper, t_args, t_ins, t_outs, t_axes,
            os.path.join(args.output_dir, "thinker.onnx"))

    # --- Talker ---
    a_wrapper = TalkerONNX(model.talker).to(device)
    a_hidden = torch.randn(1, seq_len, 768, dtype=dtype, device=device)
    a_cos = model.talker.freqs_cos[:seq_len].to(dtype=dtype, device=device)
    a_sin = model.talker.freqs_sin[:seq_len].to(dtype=dtype, device=device)
    a_args = (a_hidden, a_cos, a_sin, *[ekv] * 8)
    a_ins, a_outs, a_axes = names_and_axes(4, has_bridge=False)
    print(f"Exporting Talker ({len(model.talker.layers)} layers) ...")
    _export(a_wrapper, a_args, a_ins, a_outs, a_axes,
            os.path.join(args.output_dir, "talker.onnx"))

    print("\nDone. Test with:")
    print(f"  python scripts/eval_onnx.py --thinker_onnx {os.path.join(args.output_dir, 'thinker.onnx')} --talker_onnx {os.path.join(args.output_dir, 'talker.onnx')}")


if __name__ == "__main__":
    main()
