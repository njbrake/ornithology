#!/usr/bin/env python3
"""convert_hf_to_gguf.py — HF qwen3_5_moe safetensors -> BF16 Ornith GGUF.

This is step 1 of the pipeline in POLICY.md: produce a full-precision (BF16)
GGUF with the correct Ornith tensor names + metadata, which `ornith quantize`
then turns into the asymmetric deployable artifact. The C engine reads the GGUF
this writes (see src/ornith_gguf.c / src/ornith_gguf_write.c); the on-disk
format here is the same GGUF v3 layout, written in pure Python so the converter
has no dependency on llama.cpp's `gguf` package — only numpy + safetensors to
read the HF shards.

STATUS: runnable scaffold. The GGUF container writer and the metadata mapping
are complete and correct; the *tensor-name* mapping is concrete but provisional
for the SSM/linear-attn and MoE-expert tensors, because the exact HF parameter
names for qwen3_5_moe have not been verified against a downloaded checkpoint
(see the NOTE blocks). It needs the multi-GB weights to run end to end, which we
deliberately do not download here.

Naming target (verified against the official 9B GGUF; see POLICY.md):
  global:     token_embd.weight, output_norm.weight, output.weight
  every blk:  attn_norm, post_attention_norm, ffn_*
  full-attn:  attn_q/k/v/output, attn_q_norm, attn_k_norm
  linear/SSM: attn_qkv (fused in-proj), ssm_conv1d, ssm_a, ssm_dt.bias,
              ssm_alpha, ssm_beta, ssm_norm, ssm_out, attn_gate
  MoE:        ffn_gate_inp (router), ffn_{gate,up,down}_exps (3D, stacked over
              experts), ffn_{gate,up,down}_shexp (shared expert)

Usage:
  python3 tools/quantize/convert_hf_to_gguf.py <hf_model_dir> <out.gguf>
Then:
  ornith inspect --tensors <out.gguf>
  ornith quantize <out.gguf> <out-ornith.gguf>
"""
import json
import os
import re
import struct
import sys

# ---- GGUF constants (mirror src/ornith_gguf.c) --------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# value type tags
(T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL,
 T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64) = range(13)

# ggml tensor types (subset; match oggml_type)
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30


# ---- a minimal, dependency-free GGUF writer -----------------------------

class GGUFWriter:
    def __init__(self):
        self.kv = []        # (key, vtype, payload_bytes)
        self.tensors = []   # (name, dims(list,_ggml order_), ggml_type, raw_bytes)

    # -- metadata --
    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def add_string(self, key, val):
        self.kv.append((key, T_STRING, self._str(val)))

    def add_u32(self, key, val):
        self.kv.append((key, T_UINT32, struct.pack("<I", val)))

    def add_f32(self, key, val):
        self.kv.append((key, T_FLOAT32, struct.pack("<f", val)))

    def add_bool(self, key, val):
        self.kv.append((key, T_BOOL, struct.pack("<B", 1 if val else 0)))

    def add_arr_str(self, key, vals):
        p = struct.pack("<IQ", T_STRING, len(vals))
        for v in vals:
            p += self._str(v)
        self.kv.append((key, T_ARRAY, p))

    # -- tensors --
    def add_tensor(self, name, np_array, ggml_type):
        # GGUF stores dims in reverse of numpy's row-major shape.
        dims = list(reversed(list(np_array.shape)))
        self.tensors.append((name, dims, ggml_type, np_array.tobytes()))

    def write(self, path):
        with open(path, "wb") as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv)))
            for key, vtype, payload in self.kv:
                f.write(self._str(key))
                f.write(struct.pack("<I", vtype))
                f.write(payload)
            # tensor index with cumulative aligned offsets
            off = 0
            for name, dims, ggml_type, raw in self.tensors:
                f.write(self._str(name))
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", off))
                off = _align(off + len(raw), GGUF_ALIGNMENT)
            # data section
            _pad(f, GGUF_ALIGNMENT)
            for _name, _dims, _t, raw in self.tensors:
                f.write(raw)
                _pad(f, GGUF_ALIGNMENT)


def _align(x, a):
    return (x + a - 1) // a * a


def _pad(f, a):
    pos = f.tell()
    f.write(b"\x00" * (_align(pos, a) - pos))


# ---- HF -> Ornith tensor-name mapping -----------------------------------

# NOTE: The HF state_dict keys for qwen3_5_moe are provisional. Full-attn,
# norms and embeddings follow the standard Qwen layout; the SSM/linear-attn and
# MoE-expert keys below should be verified against a real checkpoint. The right
# *output* (Ornith GGUF) names are known (POLICY.md); only the HF side is the
# unknown, so this dict is where to fix things once a checkpoint is in hand.

_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight":         "output_norm.weight",
    "lm_head.weight":            "output.weight",
}

# per-layer: (regex on the suffix after 'model.layers.{i}.') -> ornith suffix
_LAYER = [
    (r"input_layernorm\.weight",            "attn_norm.weight"),
    (r"post_attention_layernorm\.weight",   "post_attention_norm.weight"),
    # full attention
    (r"self_attn\.q_proj\.weight",          "attn_q.weight"),
    (r"self_attn\.k_proj\.weight",          "attn_k.weight"),
    (r"self_attn\.v_proj\.weight",          "attn_v.weight"),
    (r"self_attn\.o_proj\.weight",          "attn_output.weight"),
    (r"self_attn\.q_norm\.weight",          "attn_q_norm.weight"),
    (r"self_attn\.k_norm\.weight",          "attn_k_norm.weight"),
    # linear / SSM (gated delta-net) -- provisional HF names
    (r"linear_attn\.in_proj.*\.weight",     "attn_qkv.weight"),
    (r"linear_attn\.conv1d\.weight",        "ssm_conv1d.weight"),
    (r"linear_attn\.A_log",                 "ssm_a"),
    (r"linear_attn\.dt_bias",               "ssm_dt.bias"),
    (r"linear_attn\.alpha.*\.weight",       "ssm_alpha.weight"),
    (r"linear_attn\.beta.*\.weight",        "ssm_beta.weight"),
    (r"linear_attn\.norm\.weight",          "ssm_norm.weight"),
    (r"linear_attn\.out_proj\.weight",      "ssm_out.weight"),
    (r"linear_attn\.gate.*\.weight",        "attn_gate.weight"),
    # MoE
    (r"mlp\.gate\.weight",                  "ffn_gate_inp.weight"),
    (r"mlp\.shared_expert\.gate_proj\.weight", "ffn_gate_shexp.weight"),
    (r"mlp\.shared_expert\.up_proj\.weight",   "ffn_up_shexp.weight"),
    (r"mlp\.shared_expert\.down_proj\.weight", "ffn_down_shexp.weight"),
]

# per-expert weights are stacked into one 3D tensor [n_expert, out, in]
_EXPERT = re.compile(
    r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
_EXPERT_OUT = {"gate_proj": "ffn_gate_exps.weight",
               "up_proj":   "ffn_up_exps.weight",
               "down_proj": "ffn_down_exps.weight"}


def map_global(key):
    return _GLOBAL.get(key)


def map_layer(layer_idx, suffix):
    for pat, out in _LAYER:
        if re.fullmatch(pat, suffix):
            return "blk.%d.%s" % (layer_idx, out)
    return None


# ---- metadata from config.json ------------------------------------------

def write_metadata(w, cfg):
    # MoE 35B/397B report "qwen35moe"; the dense 9B reports "qwen35".
    n_exp = cfg.get("num_experts", 0) or cfg.get("n_routed_experts", 0)
    arch = "qwen35moe" if n_exp else "qwen35"
    w.add_string("general.architecture", arch)
    w.add_string("general.name", cfg.get("_name_or_path", "Ornith-1.0"))
    p = arch  # GGUF hyperparam keys are namespaced by the arch string
    w.add_u32(p + ".context_length",     cfg.get("max_position_embeddings", 262144))
    w.add_u32(p + ".embedding_length",   cfg["hidden_size"])
    w.add_u32(p + ".block_count",        cfg["num_hidden_layers"])
    w.add_u32(p + ".attention.head_count",    cfg.get("num_attention_heads", 0))
    w.add_u32(p + ".attention.head_count_kv", cfg.get("num_key_value_heads", 0))
    w.add_u32(p + ".attention.key_length",    cfg.get("head_dim", 0))
    w.add_u32(p + ".attention.value_length",  cfg.get("head_dim", 0))
    w.add_f32(p + ".attention.layer_norm_rms_epsilon",
              float(cfg.get("rms_norm_eps", 1e-6)))
    w.add_f32(p + ".rope.freq_base", float(cfg.get("rope_theta", 1e7)))
    w.add_u32(p + ".vocab_size", cfg.get("vocab_size", 248320))
    if n_exp:
        w.add_u32(p + ".expert_count", n_exp)
        w.add_u32(p + ".expert_used_count",
                  cfg.get("num_experts_per_tok", 0))
        w.add_u32(p + ".expert_feed_forward_length",
                  cfg.get("moe_intermediate_size", 0))
        w.add_u32(p + ".expert_shared_feed_forward_length",
                  cfg.get("shared_expert_intermediate_size", 0))
    w.add_u32(p + ".full_attention_interval",
              cfg.get("full_attention_interval", 4))


# ---- driver -------------------------------------------------------------

def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src_dir, out_path = argv[1], argv[2]

    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)

    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError:
        print("error: this converter needs `numpy` and `safetensors` to read "
              "HF shards (pip install numpy safetensors).", file=sys.stderr)
        print("The name/metadata mapping above is the concrete contract; only "
              "the weight read needs these deps + the multi-GB checkpoint.",
              file=sys.stderr)
        return 1

    w = GGUFWriter()
    write_metadata(w, cfg)

    shards = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
    if not shards:
        print("error: no .safetensors found in %s" % src_dir, file=sys.stderr)
        return 1

    # gather per-expert tensors to stack after the full pass
    experts = {}   # (layer, kind) -> {expert_idx: ndarray}

    def to_bf16(arr):
        # numpy has no native bf16; emit raw 16-bit via uint16 truncation with
        # round-to-nearest-even (matches src/ornith_quant.c oq_f32_to_bf16).
        u32 = arr.astype(np.float32).view(np.uint32)
        rounding = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
        return ((u32 + rounding) >> 16).astype("<u2")

    for shard in sorted(shards):
        with safe_open(os.path.join(src_dir, shard), framework="numpy") as st:
            for key in st.keys():
                arr = st.get_tensor(key)
                g = map_global(key)
                if g:
                    w.add_tensor(g, to_bf16(arr), GGML_BF16)
                    continue
                m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
                if not m:
                    print("  skip (unmapped): %s" % key, file=sys.stderr)
                    continue
                li, suffix = int(m.group(1)), m.group(2)
                em = _EXPERT.fullmatch(suffix)
                if em:
                    kind = em.group(2)
                    experts.setdefault((li, kind), {})[int(em.group(1))] = arr
                    continue
                name = map_layer(li, suffix)
                if name:
                    w.add_tensor(name, to_bf16(arr), GGML_BF16)
                else:
                    print("  skip (unmapped): %s" % key, file=sys.stderr)

    # stack experts into 3D [n_expert, out, in] tensors
    for (li, kind), by_idx in experts.items():
        ordered = [by_idx[i] for i in sorted(by_idx)]
        stacked = np.stack(ordered, axis=0)
        name = "blk.%d.%s" % (li, _EXPERT_OUT[kind])
        w.add_tensor(name, to_bf16(stacked), GGML_BF16)

    w.write(out_path)
    print("wrote %s (%d tensors)" % (out_path, len(w.tensors)))
    print("next: ornith inspect --tensors %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
