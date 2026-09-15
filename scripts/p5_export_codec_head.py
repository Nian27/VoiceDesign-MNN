# P5: export codec_head Linear(2048->3072) as ONNX + MNN
import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
ch = gw.model.talker.codec_head
print('codec_head', tuple(ch.weight.shape), flush=True)
class CodecHead(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
    def forward(self, hidden_states):
        return self.ch(hidden_states)
core = CodecHead(ch).eval()
h = torch.randn((1,1,2048), dtype=torch.float32)
with torch.inference_mode():
    y = core(h)
print('fwd ok', tuple(y.shape), flush=True)
torch.onnx.export(core, (h,), ART + '/codec_head.onnx', input_names=['hidden_states'], output_names=['logits'], opset_version=18, dynamo=False)
print('CODEC_HEAD_EXPORT_OK', flush=True)