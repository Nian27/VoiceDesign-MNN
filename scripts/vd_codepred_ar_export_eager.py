# Re-export codepred AR with eager attention (avoids HF SDPA IsNaN mask path).
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor
cp.model.config._attn_implementation = 'eager'
print('attn impl now:', cp.model.config._attn_implementation)

class CodepredAR(torch.nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.proj = cp.small_to_mtp_projection
        self.core = cp.model
        self.lm = cp.lm_head
        self.emb = cp.model.codec_embedding
    def forward(self, past_hidden, code0_emb):
        seq = torch.cat([past_hidden, code0_emb], dim=1)
        lgs = []
        for g in range(15):
            h = self.core(inputs_embeds=self.proj(seq), use_cache=False).last_hidden_state
            lg = self.lm[g](h[:, -1:])
            lgs.append(lg)
            if g < 14:
                seq = torch.cat([seq, self.emb[g](torch.argmax(lg, dim=-1))], dim=1)
        return torch.cat(lgs, dim=1)

m = CodepredAR(cp).eval()
ph = torch.randn(1,1,2048)*0.3
c0e = torch.randn(1,1,2048)*0.3
with torch.inference_mode():
    y = m(ph, c0e)
print('fwd ok', tuple(y.shape), 'finite', bool(torch.isfinite(y).all()))
torch.save({'ph': ph, 'c0e': c0e, 'y': y}, D + '/codepred_ar/ar_torch_io.pt')
with torch.inference_mode():
    torch.onnx.export(m, (ph, c0e), D + '/codepred_ar_eager.onnx',
                      input_names=['past_hidden','code0_emb'], output_names=['logits15'],
                      opset_version=18, dynamo=False)
print('EXPORT_OK', os.path.getsize(D + '/codepred_ar_eager.onnx'))
import onnx
mm = onnx.load(D + '/codepred_ar_eager.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print('IsNaN count =', ops.get('IsNaN', 0), '| total nodes', len(mm.graph.node))
print('EAGER_EXPORT_DONE')
