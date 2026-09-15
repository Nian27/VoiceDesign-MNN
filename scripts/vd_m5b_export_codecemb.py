import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/vd_m5b'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
emb = talker.get_input_embeddings()
class CodecEmb(torch.nn.Module):
    def __init__(self, e):
        super().__init__(); self.e = e
    def forward(self, codes):        # (1,1) int64
        return self.e(codes)         # (1,1,2048)
m = CodecEmb(emb).eval()
ct = torch.tensor([[1234]])
with torch.inference_mode():
    y = m(ct)
print('codec_emb out', tuple(y.shape), 'weight', tuple(emb.weight.shape))
torch.onnx.export(m, (ct,), D + '/codec_emb_lookup.onnx', input_names=['codes'], output_names=['emb'], opset_version=18, dynamo=False)
print('size', os.path.getsize(D + '/codec_emb_lookup.onnx'))
print('CODEC_EMB_EXPORT_DONE')
