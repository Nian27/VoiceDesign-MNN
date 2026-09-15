import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
n_heads = len(cp.lm_head)
print('lm_head count:', n_heads, flush=True)
class CPStepG(torch.nn.Module):
    def __init__(self, cp, g):
        super().__init__()
        self.cp = cp
        self.g = g
    def forward(self, inputs_embeds):
        out = self.cp.model(inputs_embeds=inputs_embeds, use_cache=False)
        return self.cp.lm_head[self.g](out.last_hidden_state)
ie = torch.randn((1,1,1024), dtype=torch.float32)
for g in range(n_heads):
    core = CPStepG(cp, g).eval()
    with torch.inference_mode():
        y = core(ie)
    torch.onnx.export(
        core,
        (ie,),
        ART + '/codepred_g' + str(g) + '.onnx',
        input_names=['inputs_embeds'],
        output_names=['logits'],
        opset_version=18,
        dynamo=True,
    )
    print('g', g, 'exported', tuple(y.shape), flush=True)
print('CODEPRED_16_EXPORT_OK', flush=True)