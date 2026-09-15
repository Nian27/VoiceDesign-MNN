import os, torch, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART, exist_ok=True)
import qwen_tts.core.models.modeling_qwen3_tts as MM
m = MM.Qwen3TTSForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
talker = m.talker
code_pred = talker.code_predictor
print('talker device:', next(talker.parameters()).device, 'dtype', next(talker.parameters()).dtype, flush=True)
print('code_predictor:', type(code_pred).__name__, flush=True)
print('code_predictor forward sig:', str(inspect.signature(code_pred.forward))[:300], flush=True)
print('code_predictor model children:', [n for n,_ in code_pred.model.named_children()][:4], flush=True)
print('code_predictor num_layers', getattr(code_pred.config, 'num_hidden_layers', '?'), flush=True)
print('TALKER_PROBE_DONE', flush=True)