import os, json, time, io, numpy as np
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

MODEL = r"E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TOKENIZER = r"E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz"
RUN = r"E:/AndroidStudioProjects/qwen3tts-mnn/runs/ref_voice_design_20260824"

os.makedirs(RUN, exist_ok=True)

import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
t0 = time.time()
model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
t1 = time.time()
print("model_load_s", round(t1 - t0, 2), flush=True)

texts = [
  {"text": "我在这个夏天，只想去看看海。", "instruct": "沉稳成熟的青年男声，音域男中音，气息沉稳", "language": "Chinese"},
  {"text": "我在这个夏天，只想去看看海。", "instruct": "活泼开朗的少女声，音色明亮，语速偏快", "language": "Chinese"},
]

try:
    wavs, sr = model.generate_voice_design(text=[t["text"] for t in texts], instruct=[t["instruct"] for t in texts], language="Chinese", non_streaming_mode=True)
    print("api_ok sr=", sr, "n=", len(wavs), flush=True)
except Exception as e:
    print("api_fail", repr(e)[:300], flush=True)
    wavs, sr = None, None

t2 = time.time()
print("design_gen_s", round(t2 - t1, 2), flush=True)

meta = {"model": MODEL, "tokenizer": TOKENIZER, "load_s": t1 - t0, "gen_s": t2 - t1, "sr": sr}
if wavs is not None:
    for i, w in enumerate(wavs):
        w = np.asarray(w)
        fn = os.path.join(RUN, f"candidate_{i}.wav")
        import soundfile as sf
        sf.write(fn, w, sr)
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        mx = float(np.max(np.abs(w)))
        dur = len(w) / sr
        meta[f"cand_{i}"] = {"file": fn, "samples": int(len(w)), "dur_s": round(dur, 3), "rms": round(rms, 6), "maxabs": round(mx, 6)}
        print(f"cand {i}: dur={dur:.3f}s rms={rms:.6f} maxabs={mx:.6f}", flush=True)
with open(os.path.join(RUN, "run.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("RUN_DONE", RUN, flush=True)