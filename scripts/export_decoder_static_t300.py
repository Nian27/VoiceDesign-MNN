import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
TOK = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
os.makedirs(ART, exist_ok=True)
from qwen_tts import Qwen3TTSTokenizer
print('loading fp32...', flush=True)
tok = Qwen3TTSTokenizer.from_pretrained(TOK, dtype=torch.float32, device_map='cpu')
dec = tok.model.decoder
codes = torch.randint(0, 2048, (1, 16, 300), dtype=torch.int64)
with torch.inference_mode():
    wav = dec(codes)
print('fwd ok', tuple(wav.shape), flush=True)
dec.eval()
torch.onnx.export(
    dec,
    (codes,),
    ART + '/tokenizer_decoder_static_t300.onnx',
    input_names=['codes'],
    output_names=['waveform'],
    opset_version=18,
    dynamo=True,
)
print('DECODER_STATIC_EXPORT_OK', flush=True)