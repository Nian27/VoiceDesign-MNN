import os, json, time, torch
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
import qwen_tts.core.models.modeling_qwen3_tts as MM

MODEL = r"E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TOKENIZER = r"E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz"
ART = r"E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/onnx"
os.makedirs(ART, exist_ok=True)

def export_talker_decode_step(model):
    # talker forward signature from modeling: forward(input_ids, position_ids, ...)
    talker = model.talker
    print("talker type:", type(talker).__name__)
    # inspect forward sig
    import inspect
    print("talker.forward:", str(inspect.signature(talker.forward))[:400])
    
def export_tokenizer_decoder(tok):
    print("tokenizer dtype:", tok.model_type if hasattr(tok, "model_type") else "?")
    print("decoder module attrs:", [n for n in dir(tok) if not n.startswith("_")][:60])

print("loading model...", flush=True)
model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda", dtype=torch.bfloat16)
print("loading tokenizer...", flush=True)
tok = Qwen3TTSTokenizer.from_pretrained(TOKENIZER, device_map="cuda", dtype=torch.bfloat16)
export_talker_decode_step(model)
export_tokenizer_decoder(tok)
print("M2_PROBE_DONE", flush=True)