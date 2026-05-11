"""
Voice-only demo for ONNX MiniMindOmni — file mode with latency breakdown, or mic mode with streaming.

Usage:
    # File mode: measure latency on an audio file
    python scripts/voice_demo_onnx.py --audio ./dataset/eval_omni/audio-zh-01.mp3

    # Mic mode: live microphone → streaming text + audio output
    python scripts/voice_demo_onnx.py --mic

    # Mic mode, save output
    python scripts/voice_demo_onnx.py --mic --save_output ./output.wav
"""

import sys

# Early message — torch import takes ~60s on first load
sys.stderr.write("[voice_demo] Initializing (importing torch + onnxruntime)...")
sys.stderr.flush()

import argparse
import os
import time
import warnings
import wave

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import soundfile as sf

__package__ = "scripts"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model_omni import MiniMindOmni, OmniConfig
from model.model_minimind import precompute_freqs_cis
from dataset.omni_dataset import OmniDataset

warnings.filterwarnings("ignore")
sys.stderr.write(" done.\n")
sys.stderr.flush()

KV_HEADS, HEAD_DIM = 4, 96
T_OUT_KEY_OFFSET, A_OUT_KEY_OFFSET = 2, 1


# ===================================================================
# Lightweight ONNX Omni model (voice-focused, minimal)
# ===================================================================


class VoiceOnnxModel:
    def __init__(self, thinker_path, talker_path, device, num_threads=2):
        self.device = device
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU performance tuning
        sess_opt.intra_op_num_threads = num_threads
        sess_opt.inter_op_num_threads = 1
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opt.enable_cpu_mem_arena = True
        sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opt.add_session_config_entry("session.inter_op.allow_spinning", "0")

        self.thinker_ort = ort.InferenceSession(
            thinker_path, providers=providers, sess_options=sess_opt
        )
        self.talker_ort = ort.InferenceSession(
            talker_path, providers=providers, sess_options=sess_opt
        )

        onnx_dtype = self.talker_ort.get_outputs()[0].type
        self.use_fp16 = "float16" in onnx_dtype
        self.np_dtype = np.float16 if self.use_fp16 else np.float32
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Load PyTorch model for non-transformer components
        from transformers import AutoModelForCausalLM

        hf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minimind-3o"
        )
        self.pt_model = AutoModelForCausalLM.from_pretrained(
            hf_path, trust_remote_code=True, torch_dtype=self.torch_dtype
        )
        self.pt_model = self.pt_model.eval().to(device)

        self.embed_tokens = self.pt_model.thinker.embed_tokens
        self.lm_head = self.pt_model.thinker.lm_head  # aliased to model.lm_head
        self.audio_proj = self.pt_model.audio_proj

        t = self.pt_model.talker
        self.talker_head = t.lm_head
        self.talker_embedding = t.embed_tokens
        self.codec_proj = t.codec_proj
        self.embed_proj = t.embed_proj
        self.spk_proj = t.spk_proj
        self.text_scale = t.text_scale
        self.audio_scale = t.audio_scale

        self.config = self.pt_model.config
        self.think_end_ids = self.config.think_end_ids
        self.audio_pad_token = self.config.audio_pad_token
        self.audio_stop_token = self.config.audio_stop_token
        self.audio_spk_token = self.config.audio_spk_token

        # RoPE
        self._init_rope()

        # Move to device
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
            if mod:
                mod.to(device)
                if self.use_fp16:
                    mod.half()

        self._thinker_kv = None
        self._talker_kv = None

    def _init_rope(self):
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

    # --- KV cache ---
    def _init_kv(self, n_layers):
        cache = {}
        for i in range(n_layers):
            cache[f"past_key_{i}"] = np.zeros(
                (1, 0, KV_HEADS, HEAD_DIM), dtype=self.np_dtype
            )
            cache[f"past_value_{i}"] = np.zeros(
                (1, 0, KV_HEADS, HEAD_DIM), dtype=self.np_dtype
            )
        return cache

    def _update_kv(self, cache, outputs, offset, n_layers):
        for i in range(n_layers):
            cache[f"past_key_{i}"] = outputs[offset + 2 * i]
            cache[f"past_value_{i}"] = outputs[offset + 2 * i + 1]

    def reset_kv(self):
        self._thinker_kv = None
        self._talker_kv = None

    # --- ONNX runners ---
    def _run_thinker(self, hidden, start_pos):
        if self._thinker_kv is None:
            self._thinker_kv = self._init_kv(8)
        seq_len = hidden.shape[1]
        feed = {
            "hidden_states": hidden.to(dtype=self.torch_dtype)
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_cos": self.t_cos[start_pos : start_pos + seq_len]
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_sin": self.t_sin[start_pos : start_pos + seq_len]
            .cpu()
            .numpy()
            .astype(self.np_dtype),
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
        if self._talker_kv is None:
            self._talker_kv = self._init_kv(4)
        seq_len = hidden.shape[1]
        feed = {
            "hidden_states": hidden.to(dtype=self.torch_dtype)
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_cos": self.a_cos[start_pos : start_pos + seq_len]
            .cpu()
            .numpy()
            .astype(self.np_dtype),
            "freqs_sin": self.a_sin[start_pos : start_pos + seq_len]
            .cpu()
            .numpy()
            .astype(self.np_dtype),
        }
        for i in range(4):
            feed[f"past_key_{i}"] = self._talker_kv[f"past_key_{i}"]
            feed[f"past_value_{i}"] = self._talker_kv[f"past_value_{i}"]
        outputs = self.talker_ort.run(None, feed)
        self._update_kv(self._talker_kv, outputs, A_OUT_KEY_OFFSET, 4)
        h = torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.torch_dtype)
        return h

    # --- Modality injection ---
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

    # --- Sampling ---
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

    # --- Audio encoding ---
    def encode_audio(self, audio_inputs, audio_lens):
        if audio_inputs is None or not audio_inputs.any():
            return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.pt_model.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        vl = (
            audio_lens[batch_mask].to(valid_fbank.device)
            if audio_lens is not None
            else torch.tensor([valid_fbank.size(1)], device=valid_fbank.device)
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

    # --- Generation ---
    @torch.no_grad()
    def stream_generate(
        self,
        input_ids,
        tokenizer,
        eos_token_id=2,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.85,
        rp=1.0,
        audio_inputs=None,
        audio_lens=None,
    ):
        self.reset_kv()
        start_pos = input_ids.shape[1]
        text_finished = False
        audio_buffer = torch.full(
            (1, 8, start_pos),
            self.audio_pad_token,
            dtype=torch.long,
            device=self.device,
        )
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8

        # ---- Prefill ----
        text_ids = input_ids
        hidden = self.embed_tokens(text_ids)
        if audio_inputs is not None:
            af = self.encode_audio(audio_inputs, audio_lens)
            hidden = self._inject_audio_features(text_ids, hidden, af)

        t0 = time.time()
        h_thinker, bridge = self._run_thinker(hidden, start_pos=0)
        talker_emb = self.talker_embedding(audio_buffer)
        fused = (
            self.embed_proj(bridge) * self.text_scale
            + self.codec_proj(talker_emb) * self.audio_scale
        )
        h_talker = self._run_talker(fused, start_pos=0)
        prefill_time = time.time() - t0

        text_logits = self.lm_head(h_thinker[:, -1, :])
        text_token = self._sample_text(text_logits, input_ids, temperature, top_p, rp)
        audio_logits = self.talker_head(h_talker[:, -1, :])

        step = 0
        audio_step = -1
        for i in range(8):
            code = self.audio_pad_token
            audio_codes[i].append(code)

        audio_frames = []
        if not text_finished:
            yield text_token, None
            if text_token == eos_token_id:
                text_finished = True

        # ---- Decode loop ----
        decode_times = []
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if text_finished and all(s is not None for s in audio_stop_pos):
                break

            step += 1
            audio_step = step - 1
            input_ids = torch.cat(
                (input_ids, torch.tensor([[text_token]], device=self.device)), dim=1
            )
            new_audio = torch.full(
                (1, 8, 1), self.audio_pad_token, dtype=torch.long, device=self.device
            )
            for i in range(min(audio_step + 1, 8)):
                new_audio[0, i, 0] = audio_codes[i][-1]
            audio_buffer = torch.cat((audio_buffer, new_audio), dim=2)

            hidden = self.embed_tokens(input_ids[:, -1:])

            t_dec = time.time()
            h_thinker, bridge = self._run_thinker(
                hidden, start_pos=input_ids.shape[1] - 1
            )
            talker_emb_last = self.talker_embedding(audio_buffer[:, :, -1:])
            fused = (
                self.embed_proj(bridge) * self.text_scale
                + self.codec_proj(talker_emb_last) * self.audio_scale
            )
            h_talker = self._run_talker(fused, start_pos=input_ids.shape[1] - 1)
            decode_times.append(time.time() - t_dec)

            if text_finished:
                text_token = 0
            else:
                text_logits = self.lm_head(h_thinker[:, -1, :])
                text_token = self._sample_text(
                    text_logits, input_ids, temperature, top_p, rp
                )

            audio_logits = self.talker_head(h_talker[:, -1, :])
            for i in range(8):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    code = self._sample_audio_code(audio_logits[i], audio_codes[i])
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048:
                        audio_stop_pos[i] = len(audio_codes[i]) - 1

            audio_frame = None
            if audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active = sum(
                    1
                    for i in range(8)
                    if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i]
                )
                if active >= 8:
                    audio_frame = frame
                    audio_frames.append(frame)

            if not text_finished:
                yield text_token, audio_frame
                if text_token == eos_token_id:
                    text_finished = True
            else:
                yield None, audio_frame

        yield {
            "prefill_time": prefill_time,
            "decode_times": decode_times,
            "n_steps": step,
            "audio_frames": audio_frames,
        }, None


# ===================================================================
# Audio I/O helpers
# ===================================================================


def load_audio_file(path, target_sr=16000):
    import librosa

    samples, sr = librosa.load(path, sr=target_sr, mono=True)
    return samples.astype(np.float32), sr


def save_audio_wav(path, samples, sr=24000):
    samples = np.clip(samples / max(abs(samples).max(), 0.01), -1, 1)
    sf.write(path, samples.astype(np.float32), sr)


def decode_mimi_frames(mimi_model, frames, device):
    codes = [f for f in frames if f and len(f) == 8]
    if not codes:
        return None
    mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0).to(device)
    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
    with torch.no_grad():
        audio = mimi_model.to(device).decode(filtered).audio_values
    return audio.squeeze().float().cpu().numpy()


# ===================================================================
# File mode: latency measurement
# ===================================================================


def run_file_mode(model, tokenizer, mimi_model, args):
    samples, sr = load_audio_file(args.audio, 16000)

    # Encode audio
    t_enc = time.time()
    mel, valid_len = OmniDataset.process_audio(
        args.audio, model.pt_model.audio_processor
    )
    audio_inputs = mel.unsqueeze(0).to(args.device)
    audio_lens = torch.tensor([valid_len], device=args.device)
    prompt = model.config.audio_special_token * (valid_len or 1)
    enc_time = time.time() - t_enc

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    x = torch.tensor(
        tokenizer(inputs_text).data["input_ids"], dtype=torch.long, device=args.device
    )[None, :]

    # Generate
    gen = model.stream_generate(
        x,
        tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        audio_inputs=audio_inputs,
        audio_lens=audio_lens,
    )

    t_gen_start = time.time()
    text_tokens = []
    stats = None
    for text_token, audio_frame in gen:
        if isinstance(text_token, dict):  # final stats
            stats = text_token
        elif text_token is not None:
            text_tokens.append(text_token)
            answer = tokenizer.decode([text_token], skip_special_tokens=False)
            print(answer, end="", flush=True)
    gen_time = time.time() - t_gen_start
    print()

    # Decode audio
    t_mimi = time.time()
    audio_out = None
    if stats and stats["audio_frames"]:
        audio_out = decode_mimi_frames(mimi_model, stats["audio_frames"], args.device)
    mimi_time = time.time() - t_mimi

    # Save
    if audio_out is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(args.audio))[0] + "_onnx_out.wav"
        out_path = args.save_output or os.path.join(args.output_dir, out_name)
        save_audio_wav(out_path, audio_out)
        print(f"Saved: {out_path}")

    # Latency report
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"  Audio encoding (SenseVoice) : {enc_time * 1000:.0f} ms")
    print(f"  Total generation            : {gen_time * 1000:.0f} ms")
    if stats:
        print(f"    Prefill (Thinker+Talker)  : {stats['prefill_time'] * 1000:.0f} ms")
        avg_dec = (
            sum(stats["decode_times"]) / len(stats["decode_times"]) * 1000
            if stats["decode_times"]
            else 0
        )
        total_dec = sum(stats["decode_times"]) * 1000
        print(
            f"    Decode steps               : {stats['n_steps']} steps, avg {avg_dec:.1f} ms/step, total {total_dec:.0f} ms"
        )
        print(f"    Audio frames               : {len(stats['audio_frames'])} frames")
    print(f"  Mimi decoding               : {mimi_time * 1000:.0f} ms")
    if stats:
        ttft = stats["prefill_time"] * 1000
        print(f"\n  ⏱  Time-to-first-token : {ttft:.0f} ms")
        if audio_out is not None:
            tts = (enc_time + gen_time + mimi_time) * 1000
            print(f"  ⏱  Total voice-to-voice: {tts:.0f} ms")


# ===================================================================
# Mic mode: streaming
# ===================================================================


def run_mic_mode(model, tokenizer, mimi_model, args):
    try:
        import sounddevice as sd
    except ImportError:
        print("Please install: pip install sounddevice")
        sys.exit(1)

    # Check for input device
    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]
        if not inputs:
            print("No microphone found.")
            sys.exit(1)
        if not outputs:
            print("No speaker found for playback.")
        in_dev = inputs[0]
        print(f"Mic: {in_dev['name']} | Speaker: {outputs[0]['name'] if outputs else 'none'}")
    except Exception as e:
        print(f"No audio device: {e}")
        sys.exit(1)

    sr_target = 16000
    silence_threshold = args.vad_threshold
    silence_timeout = 1.5  # seconds of silence to end recording
    chunk_duration = 0.2  # analysis chunk size
    chunk_samples = int(chunk_duration * sr_target)
    max_record = args.max_record_secs

    print(f"Listening... (Ctrl+C to exit)")
    print()

    while True:
        # ---- Phase 1: wait for speech ----
        pre_buffer = []
        speaking = False
        audio_chunks = []

        try:
            while not speaking:
                chunk = sd.rec(chunk_samples, samplerate=sr_target, channels=1, dtype="float32", blocking=True).squeeze()
                energy = float((chunk ** 2).mean()) ** 0.5
                pre_buffer.append(chunk)
                if len(pre_buffer) > int(0.5 / chunk_duration):
                    pre_buffer.pop(0)
                if energy > silence_threshold:
                    speaking = True
                    audio_chunks = list(pre_buffer)  # keep pre-speech context

            # ---- Phase 2: record until silence ----
            silent_chunks = 0
            while True:
                chunk = sd.rec(chunk_samples, samplerate=sr_target, channels=1, dtype="float32", blocking=True).squeeze()
                energy = float((chunk ** 2).mean()) ** 0.5
                audio_chunks.append(chunk)
                if energy < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                if silent_chunks >= int(silence_timeout / chunk_duration):
                    break
                if len(audio_chunks) * chunk_duration >= max_record:
                    break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        samples = np.concatenate(audio_chunks)
        duration = len(samples) / sr_target
        if duration < 0.3:
            print("  (too short, listening...)")
            continue

        # Normalize
        max_val = abs(samples).max()
        if max_val > 0:
            samples = samples / max_val

        # Trim leading/trailing silence
        non_silent = np.where(np.abs(samples) > silence_threshold)[0]
        if len(non_silent) > 0:
            samples = samples[non_silent[0] : non_silent[-1] + 1]
        trimmed = len(samples) / sr_target
        if trimmed < 0.3:
            print("  (too short after trim, listening...)")
            continue

        print(f"[Heard {trimmed:.1f}s]", end="  ", flush=True)

        # ---- Phase 3: encode ----
        t0 = time.time()
        inputs = model.pt_model.audio_processor(
            samples, sampling_rate=sr_target, return_tensors="pt", return_attention_mask=True
        )
        mel = inputs.input_features.squeeze(0).to(args.device)
        valid_len = inputs.attention_mask.sum().item()
        audio_inputs = mel.unsqueeze(0)
        audio_lens = torch.tensor([valid_len], device=args.device)
        prompt = model.config.audio_special_token * (valid_len or 1)

        messages = [{"role": "user", "content": prompt}]
        inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        x = torch.tensor(tokenizer(inputs_text).data["input_ids"], dtype=torch.long, device=args.device)[None, :]

        # ---- Phase 4: generate ----
        gen = model.stream_generate(
            x, tokenizer, eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            audio_inputs=audio_inputs, audio_lens=audio_lens,
        )

        audio_frames = []
        print("🤖 ", end="", flush=True)
        for text_token, audio_frame in gen:
            if isinstance(text_token, dict):
                break
            if text_token is not None:
                answer = tokenizer.decode([text_token], skip_special_tokens=False)
                print(answer, end="", flush=True)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()
        gen_time = time.time() - t0

        # ---- Phase 5: decode & play ----
        if audio_frames:
            audio_out = decode_mimi_frames(mimi_model, audio_frames, args.device)
            if audio_out is not None:
                t_play = time.time()
                sd.play(audio_out, samplerate=24000, blocking=True)
                play_time = time.time() - t_play
                print(f"  [{trimmed:.1f}s → {gen_time:.1f}s (play {play_time:.1f}s)]")
            else:
                print(f"  [no audio — {gen_time:.1f}s]")
        else:
            print(f"  [text-only — {gen_time:.1f}s]")
        print()


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Voice-only ONNX demo")
    parser.add_argument("--audio", type=str, help="Input audio file path")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument(
        "--mic_duration", type=float, default=5.0, help="(Unused — VAD-based detection)"
    )
    parser.add_argument("--vad_threshold", type=float, default=0.015, help="Energy threshold for speech detection")
    parser.add_argument("--max_record_secs", type=float, default=30.0, help="Max recording before auto-stop")
    parser.add_argument("--save_output", type=str, help="Output WAV path (overrides --output_dir)")
    parser.add_argument("--output_dir", default="./output_audio", type=str)
    parser.add_argument("--thinker_onnx", default="./scripts/thinker.onnx")
    parser.add_argument("--talker_onnx", default="./scripts/talker.onnx")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--num_threads", type=int, default=2, help="ORT intra-op threads"
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--mimi_path", default="./model/mimi")
    parser.add_argument("--audio_encoder", default="./model/SenseVoiceSmall")
    parser.add_argument("--profile_memory", action="store_true")
    args = parser.parse_args()

    if not args.audio and not args.mic:
        print("Specify --audio <file> or --mic")
        sys.exit(1)

    # Init ONNX model
    print(f"Loading ONNX models ({args.device}, {args.num_threads} threads)...")
    model = VoiceOnnxModel(
        args.thinker_onnx, args.talker_onnx, args.device, args.num_threads
    )

    # Attach audio encoder
    print("Loading SenseVoice...")
    ae, ap = MiniMindOmni.load_sensevoice(args.audio_encoder)
    if ae is None:
        print("ERROR: SenseVoice not found!")
        sys.exit(1)
    object.__setattr__(model.pt_model, "audio_encoder", ae.to(args.device).float())
    object.__setattr__(model.pt_model, "audio_processor", ap)

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "minimind-3o"
        )
    )

    # Load Mimi
    from transformers import MimiModel

    mimi_model = MimiModel.from_pretrained(args.mimi_path).eval().to(args.device)

    if args.profile_memory:
        import psutil

        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss / 1024**3
        vram0 = torch.cuda.memory_allocated() / 1024**3 if args.device == "cuda" else 0
        print(f"[MEM init] RAM={rss0:.2f}G VRAM={vram0:.2f}G")

    if args.audio:
        run_file_mode(model, tokenizer, mimi_model, args)
    elif args.mic:
        run_mic_mode(model, tokenizer, mimi_model, args)


if __name__ == "__main__":
    main()
