import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
print('cp children:', [n for n,_ in cp.named_children()], flush=True)
# 单步 decode: inputs_embeds (1,1,1024) -> logits (1,1,2048)
class CPStep(torch.nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp
    def forward(self, inputs_embeds):
        out = self.cp.model(inputs_embeds=inputs_embeds, use_cache=False)
        return self.cp.lm_head(out.last_hidden_state)
core = CPStep(cp).eval()
ie = torch.randn((1,1,1024), dtype=torch.float32)
with torch.inference_mode():
    y = core(ie)
print('fwd ok', tuple(y.shape), flush=True)
torch.onnx.export(
    core,
    (ie,),
    ART + '/codepred_step.onnx',
    input_names=['inputs_embeds'],
    output_names=['logits'],
    opset_version=18,
    dynamo=True,
)
print('CODEPRED_EXPORT_OK', flush=True)