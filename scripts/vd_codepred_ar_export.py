# Export single-graph codepred AR using the OFFICIAL model (no manual reimpl).
# inputs: past_hidden(1,1,2048), code0_emb(1,1,2048)
# outputs: logits(1,15,2048), codes(1,15)
import os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor

class CodepredAR(torch.nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.proj = cp.small_to_mtp_projection
        self.core = cp.model
        self.lm = cp.lm_head
        self.emb = cp.model.codec_embedding
    def forward(self, past_hidden, code0_emb):
        seq = torch.cat([past_hidden, code0_emb], dim=1)      # (1,2,2048)
        lgs = []
        for g in range(15):
            x = self.proj(seq)                                 # (1,n,1024)
            h = self.core(inputs_embeds=x, use_cache=False).last_hidden_state
            lg = self.lm[g](h[:, -1:])                         # (1,1,2048)
            lgs.append(lg)
            if g < 14:
                c = torch.argmax(lg, dim=-1)                   # (1,1) int64
                seq = torch.cat([seq, self.emb[g](c)], dim=1)
        return torch.cat(lgs, dim=1)                           # (1,15,2048)

m = CodepredAR(cp).eval()
ph = torch.randn(1,1,2048)*0.3
c0e = torch.randn(1,1,2048)*0.3
with torch.inference_mode():
    y = m(ph, c0e)
print('fwd ok', tuple(y.shape), 'finite', bool(torch.isfinite(y).all()))
torch.save({'ph': ph, 'c0e': c0e, 'y': y}, D + '/codepred_ar/ar_torch_io.pt')
try:
    with torch.inference_mode():
        torch.onnx.export(m, (ph, c0e), D + '/codepred_ar_fp32.onnx',
                          input_names=['past_hidden','code0_emb'],
                          output_names=['logits15'],
                          opset_version=18, dynamo=False)
    print('EXPORT_LEGACY_OK', os.path.getsize(D + '/codepred_ar_fp32.onnx'))
except Exception as e:
    print('legacy export failed:', str(e)[:300])
    with torch.inference_mode():
        torch.onnx.export(m, (ph, c0e), D + '/codepred_ar_fp32.onnx',
                          input_names=['past_hidden','code0_emb'],
                          output_names=['logits15'],
                          opset_version=18, dynamo=True)
    print('EXPORT_DYNAMO_OK', os.path.getsize(D + '/codepred_ar_fp32.onnx'))
print('AR_EXPORT_ATTEMPT_DONE')
