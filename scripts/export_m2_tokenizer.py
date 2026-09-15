import os, json, time, torch, numpy as np, inspect
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
TOK = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-Tokenizer-12Hz'
from qwen_tts import Qwen3TTSTokenizer
tok = Qwen3TTSTokenizer.from_pretrained(TOK, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
print('tok methods:', [n for n in dir(tok) if not n.startswith('_')][:60], flush=True)
print('decode sig:', str(inspect.signature(tok.decode))[:400], flush=True)
print('encode sig:', str(inspect.signature(tok.encode))[:400], flush=True)
print('DONE_PROBE', flush=True)