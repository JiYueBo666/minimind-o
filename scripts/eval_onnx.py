"""
MiniMind-O voice evaluation using ONNX Runtime + Python-side glue logic.

Usage:
    python scripts/eval_onnx.py --thinker_onnx ./scripts/thinker.onnx --talker_onnx ./scripts/talker.onnx
    python scripts/eval_onnx.py --thinker_onnx ./scripts/thinker_int8.onnx --talker_onnx ./scripts/talker_int8.onnx --device cpu

Prerequisite: run scripts/onnx_export.py first to generate the ONNX files.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import soundfile as sf
from transformers import AutoTokenizer, MimiModel

__package__ = "scripts"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model_omni import MiniMindOmni, OmniConfig
from model.model_minimind import precompute_freqs_cis
from dataset.omni_dataset import OmniDataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# ONNX inference model
# ---------------------------------------------------------------------------

KV_HEADS = 4
HEAD_DIM = 96

# Output ordering from ONNX exporter:
#   thinker: h_thinker, bridge_states, pk0, pv0, ..., pk7, pv7  (18 outputs)
#   talker:  h_talker,  pk0, pv0, pk1, pv1, pk2, pv2, pk3, pv3 (9 outputs)
T_OUT_KEY_OFFSET = 2  # thinker outputs: first KV at index 2
A_OUT_KEY_OFFSET = 1  # talker outputs:  first KV at index 1


class OnnxOmniModel:
    """MiniMindOmni for inference, with transformer layers running on ONNX Runtime."""

    def __init__(
        self,
        thinker_path,
        talker_path,
        load_from,
        checkpoint,
        device,
        hidden_size,
        num_layers,
        use_moe,
    ):
        self.device = device
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # ONNX sessions
        self.thinker_ort = ort.InferenceSession(
            thinker_path, providers=providers, sess_options=sess_opt
        )
        self.talker_ort = ort.InferenceSession(
            talker_path, providers=providers, sess_options=sess_opt
        )

        # Determine precision from ONNX model
        onnx_dtype = self.talker_ort.get_outputs()[
            0
        ].type  # e.g. "tensor(float)" or "tensor(float16)"
        self.use_fp16 = "float16" in onnx_dtype
        self.np_dtype = np.float16 if self.use_fp16 else np.float32
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Load PyTorch model for non-transformer components
        self._load_pytorch_model(
            load_from, checkpoint, hidden_size, num_layers, use_moe
        )

        # Config (must be set before RoPE init)
        self.config = self.pt_model.config
        self.think_end_ids = self.config.think_end_ids
        self.audio_pad_token = self.config.audio_pad_token
        self.audio_stop_token = self.config.audio_stop_token
        self.audio_spk_token = self.config.audio_spk_token

        # RoPE buffers
        self._init_rope_buffers()

        # Move heads/projectors to device
        for name in [
            "audio_proj",
            "codec_proj",
            "embed_proj",
            "spk_proj",
            "talker_head",
            "talker_embedding",
            "embed_tokens",
            "lm_head",
        ]:
            mod = getattr(self, name, None)
            if mod is not None:
                mod.to(device)
                if self.use_fp16:
                    mod.half()

        # Cached KV tensors (reused each generation to avoid alloc)
        self._thinker_kv = None
        self._talker_kv = None

    def _load_pytorch_model(
        self, load_from, checkpoint, hidden_size, num_layers, use_moe
    ):
        if load_from == "model":
            config = OmniConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                use_moe=bool(use_moe),
                flash_attn=False,
            )
            model = MiniMindOmni(
                config,
                audio_encoder_path="./model/SenseVoiceSmall",
                vision_model_path=None,
            )
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(state, strict=False)
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                load_from, trust_remote_code=True, dtype=self.torch_dtype
            )
            model = model.eval().to(self.device)
        self.pt_model = model

        # Extract non-transformer components
        self.embed_tokens = model.thinker.embed_tokens
        self.lm_head = model.thinker.lm_head  # actually model.lm_head (aliased)
        self.audio_proj = model.audio_proj
        self.vision_proj = model.vision_proj

        # Talker components
        t = model.talker
        self.talker_lm_head = t.lm_head  # TalkerHead (8 parallel heads)
        self.talker_embedding = t.embed_tokens  # TalkerEmbedding
        self.codec_proj = t.codec_proj
        self.embed_proj = t.embed_proj
        self.spk_proj = t.spk_proj
        self.text_scale = t.text_scale
        self.audio_scale = t.audio_scale

    def _init_rope_buffers(self):
        thinker_cfg = self.config
        talker_cfg = self.pt_model.talker.talker_config
        end = thinker_cfg.max_position_embeddings

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=thinker_cfg.head_dim,
            end=end,
            rope_base=thinker_cfg.rope_theta,
            rope_scaling=thinker_cfg.rope_scaling,
        )
        self.t_cos = freqs_cos.to(dtype=self.torch_dtype, device=self.device)
        self.t_sin = freqs_sin.to(dtype=self.torch_dtype, device=self.device)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=talker_cfg.head_dim,
            end=end,
            rope_base=talker_cfg.rope_theta,
            rope_scaling=talker_cfg.rope_scaling,
        )
        self.a_cos = freqs_cos.to(dtype=self.torch_dtype, device=self.device)
        self.a_sin = freqs_sin.to(dtype=self.torch_dtype, device=self.device)

    # -----------------------------------------------------------------------
    # KV cache helpers
    # -----------------------------------------------------------------------

    def _init_kv_cache(self, n_layers):
        cache = {}
        for i in range(n_layers):
            cache[f"past_key_{i}"] = np.zeros(
                (1, 0, KV_HEADS, HEAD_DIM), dtype=self.np_dtype
            )
            cache[f"past_value_{i}"] = np.zeros(
                (1, 0, KV_HEADS, HEAD_DIM), dtype=self.np_dtype
            )
        return cache

    def _update_kv(self, cache, outputs, key_offset, n_layers):
        for i in range(n_layers):
            cache[f"past_key_{i}"] = outputs[key_offset + 2 * i]
            cache[f"past_value_{i}"] = outputs[key_offset + 2 * i + 1]

    # -----------------------------------------------------------------------
    # ONNX runner helpers
    # -----------------------------------------------------------------------

    def _run_thinker(self, hidden, start_pos):
        """Run Thinker ONNX. hidden: (1, seq, 768) torch tensor. Returns (h, bridge, kv_dict)."""
        if self._thinker_kv is None:
            self._thinker_kv = self._init_kv_cache(8)

        seq_len = hidden.shape[1]
        cos = self.t_cos[start_pos : start_pos + seq_len]
        sin = self.t_sin[start_pos : start_pos + seq_len]

        feed = {
            "hidden_states": hidden.to(dtype=self.torch_dtype)
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_cos": cos.cpu().numpy().astype(self.np_dtype),
            "freqs_sin": sin.cpu().numpy().astype(self.np_dtype),
        }
        for i in range(8):
            feed[f"past_key_{i}"] = self._thinker_kv[f"past_key_{i}"]
            feed[f"past_value_{i}"] = self._thinker_kv[f"past_value_{i}"]

        outputs = self.thinker_ort.run(None, feed)
        self._update_kv(self._thinker_kv, outputs, T_OUT_KEY_OFFSET, 8)

        h = torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.torch_dtype)
        bridge = torch.from_numpy(outputs[1]).to(
            device=self.device, dtype=self.torch_dtype
        )
        return h, bridge

    def _run_talker(self, hidden, start_pos):
        """Run Talker ONNX. hidden: (1, seq, 768) torch tensor. Returns (h, kv_dict)."""
        if self._talker_kv is None:
            self._talker_kv = self._init_kv_cache(4)

        seq_len = hidden.shape[1]
        cos = self.a_cos[start_pos : start_pos + seq_len]
        sin = self.a_sin[start_pos : start_pos + seq_len]

        feed = {
            "hidden_states": hidden.to(dtype=self.torch_dtype)
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_cos": cos.cpu().numpy().astype(self.np_dtype),
            "freqs_sin": sin.cpu().numpy().astype(self.np_dtype),
        }
        for i in range(4):
            feed[f"past_key_{i}"] = self._talker_kv[f"past_key_{i}"]
            feed[f"past_value_{i}"] = self._talker_kv[f"past_value_{i}"]

        outputs = self.talker_ort.run(None, feed)
        self._update_kv(self._talker_kv, outputs, A_OUT_KEY_OFFSET, 4)

        h = torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.torch_dtype)
        return h

    def reset_kv_cache(self):
        self._thinker_kv = None
        self._talker_kv = None

    # -----------------------------------------------------------------------
    # Modality injection (Python-side, mirrors original model_omni.py)
    # -----------------------------------------------------------------------

    def _inject_audio_features(self, tokens, h, audio_feats):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        for b in range(h.size(0)):
            hb, seq = h[b], tokens[b].tolist()
            af = audio_feats[b] if audio_feats[b] is not None else None
            i = 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat(
                            (hb[:start], af[:inject_len], hb[start + inject_len :]),
                            dim=0,
                        )
                        af = None
                else:
                    i += 1
            if b == 0:
                out = hb.unsqueeze(0)
            else:
                out = torch.cat((out, hb.unsqueeze(0)), dim=0)
        return out

    # -----------------------------------------------------------------------
    # Audio encoding
    # -----------------------------------------------------------------------

    def encode_audio(self, audio_inputs, audio_lens):
        if audio_inputs is None or self.pt_model.audio_encoder is None:
            return None
        if not audio_inputs.any():
            return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.pt_model.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        vl = (
            audio_lens[batch_mask].to(valid_fbank.device)
            if audio_lens is not None
            else torch.tensor(
                [valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device
            )
        )
        with torch.no_grad():
            emb, _ = self.pt_model.audio_encoder(valid_fbank, vl)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = []
        for i in range(emb.size(0)):
            elen = max(1, min(vl[i].item(), emb.size(1)))
            emb_list.append(
                self.audio_proj(emb[i, :elen].unsqueeze(0).to(proj_dtype)).squeeze(0)
            )
        if batch_mask.all():
            return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    # -----------------------------------------------------------------------
    # Generation loop
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.85,
        rp=1.0,
        return_audio_codes=False,
        audio_inputs=None,
        audio_lens=None,
        ref_codes=None,
        spk_emb=None,
    ):
        self.reset_kv_cache()
        start_pos = input_ids.shape[1]
        text_finished = False

        # Prepare audio_buffer
        audio_buffer = torch.full(
            (1, 8, start_pos),
            self.audio_pad_token,
            dtype=torch.long,
            device=self.device,
        )

        # Fill reference codes + spk token
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0
        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            audio_buffer[:, :, fill_start:fill_end] = ref_codes[
                :, :, -(fill_end - fill_start) :
            ]
        if spk_emb is not None and fill_start > 0:
            audio_buffer[:, :, fill_start - 1] = self.audio_spk_token

        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        think_end_step = None
        generated_tokens = [] if open_thinking else None

        # ---- Prefill (step 0) ----
        text_ids = input_ids  # (1, prompt_len)
        hidden = self.embed_tokens(text_ids)

        # Inject modalities
        if audio_inputs is not None:
            af = self.encode_audio(audio_inputs, audio_lens)
            hidden = self._inject_audio_features(text_ids, hidden, af)

        # Thinker forward
        h_thinker, bridge = self._run_thinker(hidden, start_pos=0)

        # Talker embedding + fusion
        talker_emb = self.talker_embedding(audio_buffer)
        if spk_emb is not None:
            spk_mask = (audio_buffer[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(
                spk_mask,
                self.spk_proj(spk_emb.to(self.torch_dtype)).unsqueeze(1),
                talker_emb,
            )

        fused = (
            self.embed_proj(bridge) * self.text_scale
            + self.codec_proj(talker_emb) * self.audio_scale
        )
        h_talker = self._run_talker(fused, start_pos=0)

        # Sample first text token + audio codes
        text_logits = self.lm_head(h_thinker[:, -1, :])
        text_token = self._sample_text(text_logits, input_ids, temperature, top_p, rp)
        audio_logits = self.talker_lm_head(h_talker[:, -1, :])

        step = 0
        audio_step = -1

        for i in range(8):
            code = self.audio_pad_token
            audio_codes[i].append(code)

        audio_frame = None
        if not text_finished:
            yield text_token, audio_frame
            if text_token == eos_token_id:
                text_finished = True
        else:
            yield None, audio_frame

        # ---- Decode loop ----
        while input_ids.shape[1] < start_pos + max_new_tokens:
            # Check termination
            if text_finished and all(s is not None for s in audio_stop_pos):
                break

            step += 1
            audio_step = step - 1
            if generated_tokens is not None:
                if not think_end_step and generated_tokens[
                    -len(self.think_end_ids) :
                ] == list(self.think_end_ids):
                    think_end_step = step + 2
                audio_step = -1 if think_end_step is None else step - think_end_step

            # Append text token
            input_ids = torch.cat(
                (input_ids, torch.tensor([[text_token]], device=self.device)), dim=1
            )
            # Append audio frame placeholder
            new_audio = torch.full(
                (1, 8, 1), self.audio_pad_token, dtype=torch.long, device=self.device
            )
            for i in range(min(audio_step + 1, 8)):
                new_audio[0, i, 0] = audio_codes[i][-1]
            audio_buffer = torch.cat((audio_buffer, new_audio), dim=2)

            # Embed single token
            hidden = self.embed_tokens(input_ids[:, -1:])  # (1, 1, 768)

            # Thinker decode step
            h_thinker, bridge = self._run_thinker(
                hidden, start_pos=input_ids.shape[1] - 1
            )

            # Talker embedding + fusion (single frame)
            talker_emb_last = self.talker_embedding(audio_buffer[:, :, -1:])
            if spk_emb is not None:
                spk_mask_last = (
                    audio_buffer[:, 0, -1:] == self.audio_spk_token
                ).unsqueeze(-1)
                talker_emb_last = torch.where(
                    spk_mask_last,
                    self.spk_proj(spk_emb.to(self.torch_dtype)).unsqueeze(1),
                    talker_emb_last,
                )

            fused = (
                self.embed_proj(bridge) * self.text_scale
                + self.codec_proj(talker_emb_last) * self.audio_scale
            )
            h_talker = self._run_talker(fused, start_pos=input_ids.shape[1] - 1)

            # Sample
            if text_finished:
                text_token = self.audio_pad_token if step == 1 else 0
            else:
                text_logits = self.lm_head(h_thinker[:, -1, :])
                text_token = self._sample_text(
                    text_logits, input_ids, temperature, top_p, rp
                )
                if generated_tokens is not None:
                    generated_tokens.append(text_token)

            audio_logits = self.talker_lm_head(h_talker[:, -1, :])
            for i in range(8):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    code = self._sample_audio_code(audio_logits[i], audio_codes[i])
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            # Build audio frame
            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active = sum(
                    1
                    for i in range(8)
                    if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i]
                )
                if active >= 8:
                    audio_frame = frame

            if not text_finished:
                if voice_only:
                    yield None, audio_frame
                else:
                    yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame

    # -----------------------------------------------------------------------
    # Sampling helpers
    # -----------------------------------------------------------------------

    def _sample_text(self, logits, input_ids, temperature, top_p, rp):
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        elif logits.dim() == 2:
            logits = logits[0, :] if logits.size(0) == 1 else logits[-1, :]
        logits = logits.clone().float() / (temperature + 1e-9)
        if rp != 1.0:
            seen = list(set(input_ids[0].tolist()))
            score = logits[seen]
            logits[seen] = torch.where(score > 0, score / rp, score * rp)
        if top_p and top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            logits[sorted_i[mask]] = -float("Inf")
        return torch.multinomial(F.softmax(logits, dim=-1), 1).item()

    def _sample_audio_code(self, logits_i, code_history):
        if logits_i.dim() == 3:
            logits_i = logits_i[0, -1, :]
        elif logits_i.dim() == 2:
            logits_i = logits_i[0, :] if logits_i.size(0) == 1 else logits_i[-1, :]
        logits_i = logits_i.clone().float() / 0.2
        for prev_code in code_history[-3:]:
            score = logits_i[prev_code]
            logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
        top50_vals, top50_idx = logits_i.topk(50)
        return top50_idx[torch.multinomial(F.softmax(top50_vals, dim=-1), 1)].item()


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def frames_to_mimi(frames):
    codes = [f for f in frames if f and len(f) == 8]
    if not codes:
        return None
    return torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)


def decode_mimi(mimi_model, mimi_codes):
    if mimi_codes is None or mimi_codes.numel() == 0:
        return None
    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
    filtered = filtered.to(next(mimi_model.parameters()).device)
    with torch.no_grad():
        audio = mimi_model.decode(filtered).audio_values
    return audio.squeeze().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Model init & eval
# ---------------------------------------------------------------------------


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = OnnxOmniModel(
        thinker_path=args.thinker_onnx,
        talker_path=args.talker_onnx,
        load_from=args.load_from,
        checkpoint=args.checkpoint,
        device=args.device,
        hidden_size=args.hidden_size,
        num_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    # Attach Mimi
    model.mimi_model = MimiModel.from_pretrained(args.mimi_path).eval()
    return model, tokenizer


def eval_sample(
    model,
    tokenizer,
    args,
    idx,
    prompt,
    audio_inputs,
    output_name,
    pixel_values=None,
    history=None,
    audio_lens=None,
    ref_codes=None,
    spk_emb=None,
):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(args.open_thinking),
    )
    x = torch.tensor(
        tokenizer(inputs_text).data["input_ids"], dtype=torch.long, device=args.device
    )[None, ...]

    audio_frames = []
    with torch.no_grad():
        res_y = model.stream_generate(
            x,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            return_audio_codes=True,
            open_thinking=bool(args.open_thinking),
            audio_inputs=audio_inputs,
            audio_lens=audio_lens,
            pixel_values=pixel_values,
            ref_codes=ref_codes,
            spk_emb=spk_emb,
        )
        print("📒 [Thinker]: ", end="", flush=True)
        history_idx = 0
        for y, audio_frame in res_y:
            if y is not None:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if answer and answer[-1] != "�":
                    print(answer[history_idx:], end="", flush=True)
                    history_idx = len(answer)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()

        if audio_frames:
            print(f"🎹 [Talker]: {len(audio_frames)} frames", end=" ")
            if args.decode_audio:
                try:
                    codes = [f for f in audio_frames if f and len(f) == 8]
                    if not codes:
                        print("⚠️  No valid Mimi codes, skipping save.")
                        return
                    mimi_codes = (
                        torch.tensor(codes, dtype=torch.long)
                        .T.unsqueeze(0)
                        .to(args.device)
                    )
                    filtered = torch.where(
                        mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes
                    )
                    audio = model.mimi_model.decode(filtered).audio_values
                    output_path = os.path.join(args.output_dir, output_name)
                    wav_path = output_path.rsplit(".", 1)[0] + ".wav"
                    sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)
                    AudioSegment.from_wav(wav_path).export(
                        output_path, format="mp3", bitrate="64k"
                    )
                    os.remove(wav_path)
                    print(f"| Audio decoded to: {output_path}")
                except Exception as e:
                    print(f"⚠️  Audio save failed: {e}")
            else:
                print("(decode_audio=off)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O ONNX Eval")
    parser.add_argument("--thinker_onnx", default="./scripts/thinker.onnx")
    parser.add_argument("--talker_onnx", default="./scripts/talker.onnx")
    parser.add_argument(
        "--load_from",
        default="minimind-3o",
        help="HuggingFace model dir for tokenizer + heads",
    )
    parser.add_argument("--checkpoint", default="./out/sft_omni_768.pth")
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--output_dir", default="./output_audio/", type=str)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str
    )
    parser.add_argument("--audio_dir", default="./dataset/eval_omni/", type=str)
    parser.add_argument("--image_dir", default="./dataset/eval_omni/", type=str)
    parser.add_argument("--open_thinking", default=0, type=int)
    parser.add_argument("--decode_audio", default=1, type=int)
    parser.add_argument("--mode", default="0", type=str)
    parser.add_argument("--prompt_lang", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("--mimi_path", default="./model/mimi", type=str)
    parser.add_argument("--audio_encoder", default="./model/SenseVoiceSmall", type=str)
    parser.add_argument(
        "--vision_model", default="./model/siglip2-base-p32-256-ve", type=str
    )
    args = parser.parse_args()

    modes = set(args.mode.replace(",", "").replace("-1", "012345"))
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = init_model(args)
    random.seed(int(time.time()) % 31415926)
    torch.manual_seed(random.randint(0, 2**31))

    # Attach audio_encoder + audio_processor for audio modes
    ae, ap = MiniMindOmni.load_sensevoice(args.audio_encoder)
    if ae is not None:
        object.__setattr__(model.pt_model, "audio_encoder", ae.to(args.device).float())
        object.__setattr__(model.pt_model, "audio_processor", ap)

    # Attach vision_encoder + vision_processor for image modes
    ve, vp = MiniMindOmni.load_vision(args.vision_model)
    object.__setattr__(model.pt_model, "vision_encoder", ve)
    object.__setattr__(model.pt_model, "vision_processor", vp)

    # ---- Mode handlers ----

    if "0" in modes:
        print(
            "\n\n==================== text -> {text, audio} (ONNX) ===================="
        )
        test_prompts_en = [
            "Tell me an interesting fact about space.",
            "How do I make a cup of coffee?",
            "What's the weather like today?",
            "Will it rain tomorrow?",
            "Tell me a joke.",
        ]
        test_prompts_zh = [
            "告诉我一个关于太空的有趣事实。",
            "如何制作一杯咖啡？",
            "今天的天气怎么样？",
            "明天会下雨吗？",
            "给我讲个笑话吧",
        ]
        test_prompts = [
            test_prompts_en,
            test_prompts_zh,
            test_prompts_en + test_prompts_zh,
        ][args.prompt_lang]
        for idx, prompt in enumerate(test_prompts):
            print(f"\n📝 [text-{idx + 1}]: {prompt}")
            eval_sample(
                model, tokenizer, args, idx, prompt, None, f"text-onnx-{idx:02d}.mp3"
            )

    if "1" in modes:
        print(
            "\n\n==================== multi-turn -> {text, audio} (ONNX) ===================="
        )
        multi_tests_en = [
            {
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ],
                "prompt": "I want to find something to do. Do you have any suggestions?",
            },
        ]
        multi_tests_zh = [
            {
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"},
                ],
                "prompt": "我想找点事做，你有什么建议吗？",
            },
        ]
        multi_tests = [multi_tests_en, multi_tests_zh, multi_tests_en + multi_tests_zh][
            args.prompt_lang
        ]
        for idx, test in enumerate(multi_tests):
            print(f"\n💬 [multi-{idx + 1}]")
            for msg in test["history"]:
                print(f'   {msg["role"]}: {msg["content"]}')
            print(f'   user: {test["prompt"]}')
            eval_sample(
                model,
                tokenizer,
                args,
                idx,
                test["prompt"],
                None,
                f"multi-onnx-{idx:02d}.mp3",
                history=test["history"],
            )

    if "2" in modes:
        print(
            "\n\n==================== audio -> {text, audio} (ONNX) ===================="
        )
        audio_files_en = sorted(
            [
                f
                for f in os.listdir(args.audio_dir)
                if f.startswith("audio-en-") and f.lower().endswith((".mp3", ".wav"))
            ]
        )
        audio_files_zh = sorted(
            [
                f
                for f in os.listdir(args.audio_dir)
                if f.startswith("audio-zh-") and f.lower().endswith((".mp3", ".wav"))
            ]
        )
        audio_files = [audio_files_en, audio_files_zh, audio_files_en + audio_files_zh][
            args.prompt_lang
        ]
        for idx, audio_file in enumerate(audio_files):
            print(f"\n🎤 [audio-{idx + 1}]: {audio_file}")
            mel, valid_len = OmniDataset.process_audio(
                os.path.join(args.audio_dir, audio_file), model.pt_model.audio_processor
            )
            audio_inputs = mel.unsqueeze(0).to(args.device)
            audio_lens = torch.tensor([valid_len], device=args.device)
            prompt = model.config.audio_special_token * (valid_len or 1)
            eval_sample(
                model,
                tokenizer,
                args,
                idx,
                prompt,
                audio_inputs,
                f"audio-onnx-{idx:02d}.mp3",
                audio_lens=audio_lens,
            )

    if "3" in modes:
        print(
            "\n\n==================== clone voice -> {text, audio} (ONNX) ===================="
        )
        clone_prompts_zh = ["你好，请介绍一下你自己。", "今天天气怎么样？"]
        clone_prompts_en = [
            "Hello, please introduce yourself.",
            "What's the weather like today?",
        ]
        clone_prompts = [
            clone_prompts_en,
            clone_prompts_zh,
            clone_prompts_en + clone_prompts_zh,
        ][args.prompt_lang]
        voices_pt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model",
            "speaker",
            "voices_unseen.pt",
        )
        voices = [("default", None, None)]
        if os.path.exists(voices_pt):
            voice_data = torch.load(voices_pt, map_location=args.device)
            for speaker, v in sorted(voice_data.items()):
                rc = v["ref_codes"].unsqueeze(0).to(args.device)
                se = (
                    v["spk_emb"].half().unsqueeze(0).to(args.device)
                    if "spk_emb" in v
                    else None
                )
                voices.append((speaker, rc, se))
        for speaker, rc, se in voices:
            info = f'ref_codes: {rc.shape[2] if rc is not None else "-"} frames, spk_emb: {"+" if se is not None else "-"}'
            print(f"\n🎵 [clone: {speaker}] {info}")
            for idx, prompt in enumerate(clone_prompts):
                print(f"  📝 [text-{idx + 1}]: {prompt}")
                history = [
                    {
                        "role": "system",
                        "content": "你是一个专业的语音助手，请用给定的音色风格来回答用户的问题。",
                    }
                ]
                eval_sample(
                    model,
                    tokenizer,
                    args,
                    idx,
                    prompt,
                    None,
                    f"clone-onnx-{speaker}-{idx:02d}.mp3",
                    ref_codes=rc,
                    history=history,
                    spk_emb=se,
                )

    # Modes 4 and 5 depend on load_vision being functional (currently disabled)
    if "4" in modes or "5" in modes:
        if (
            model.pt_model.vision_encoder is None
            or model.pt_model.vision_processor is None
        ):
            print(
                "\n⚠️  Vision disabled (load_vision returns None). Skipping mode 4/5."
            )
        else:
            if "4" in modes:
                print(
                    "\n\n==================== image -> {text, audio} (ONNX) ===================="
                )
                image_files = sorted(
                    [
                        f
                        for f in os.listdir(args.image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
                for idx, image_file in enumerate(image_files):
                    print(f"\n🖼️ [image-{idx + 1}]: {image_file}")
                    image = Image.open(
                        os.path.join(args.image_dir, image_file)
                    ).convert("RGB")
                    pixel_values = {
                        k: v.to(args.device)
                        for k, v in model.pt_model.vision_processor(
                            images=image, return_tensors="pt"
                        ).items()
                    }
                    prompts = [
                        ["Please describe this image."],
                        ["请描述这张图片"],
                        ["Please describe this image.", "请描述这张图片"],
                    ][args.prompt_lang]
                    for lang_idx, prompt_text in enumerate(prompts):
                        prompt = (
                            prompt_text
                            + "\n\n"
                            + model.config.image_special_token
                            * model.config.image_token_len
                        )
                        eval_sample(
                            model,
                            tokenizer,
                            args,
                            idx,
                            prompt,
                            None,
                            f"image-onnx-{idx:02d}.mp3",
                            pixel_values=pixel_values,
                        )

            if "5" in modes:
                print(
                    "\n\n==================== text+audio+image -> {text, audio} (ONNX) ===================="
                )
                # (Same as mode 4 logic but with audio)
                print("(mode 5 not yet implemented for ONNX)")

    print("\nDone.")


if __name__ == "__main__":
    main()
