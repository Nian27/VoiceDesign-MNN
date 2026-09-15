import os, torch, inspect, time
os.environ['HF_HUB_OFFLINE'] = '1'
TOK = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m2'
os.makedirs(ART, exist_ok=True)
from qwen_tts import Qwen3TTSTokenizer
t0 = time.time()
print('loading...', flush=True)
tok = Qwen3TTSTokenizer.from_pretrained(TOK, dtype=torch.float32, device_map='cuda' if torch.cuda.is_available() else 'cpu')
print('load_s', round(time.time() - t0, 2), flush=True)
dec = tok.model.decoder
print('decoder:', type(dec).__name__, flush=True)
# codes contract: (B, Q=16, T) frames
codes = torch.randint(0, 2048, (1, 16, 300), dtype=torch.long, device=dec.device)
with torch.inference_mode():
    wav = dec(codes)
print('wav shape:', tuple(wav.shape), wav.dtype, flush=True)
print('wav finite:', bool(torch.isfinite(wav).all()), flush=True)
print('wav maxabs:', float(wav.abs().max()), flush=True)
with torch.inference_mode():
    wav_chunked = dec.chunked_decode(codes, chunk_size=300, left_context_size=25)
print('chunked wav shape:', tuple(wav_chunked.shape), flush=True)
print('chunked==full:', bool(torch.allclose(wav, wav_chunked, atol=1e-4)), flush=True)
dec.eval()
try:
    torch.onnx.export(
        dec,
        (codes,),
        os.path.join(ART, 'tokenizer_decoder.onnx'),
        input_names=['codes'],
        output_names=['waveform'],
        dynamic_axes={'codes': {2: 'frames'}, 'waveform': {2: 'samples'}},
        opset_version=17,
        do_constant_folding=True,
    )
    print('ONNX_EXPORT_OK', flush=True)
except Exception as e:
    print('ONNX_EXPORT_FAIL', repr(e)[:800], flush=True)
torch.save({'codes': codes.cpu(), 'wav': wav.cpu()}, os.path.join(ART, 'golden_io.pt'))
print('DONE', flush=True)