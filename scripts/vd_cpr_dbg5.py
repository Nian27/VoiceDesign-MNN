# 5-layer debug: expose hidden after EACH layer of the prefill path
import os, math, numpy as np, torch, onnxruntime as ort
os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(1)
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'; CR = D + '/vd_cpr'
from qwen_tts import Qwen3TTSModel
import qwen_tts.core.models.modeling_qwen3_tts as MF
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
cp = gw.model.talker.code_predictor; core = cp.model
NL, NH, NKV, HD, CAP = 5, 16, 8, 128, 16
SCALE = 1.0/math.sqrt(HD)
def rh(x):
    x1 = x[..., : x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
class P5(torch.nn.Module):
    def __init__(self, cp, core):
        super().__init__(); self.cp = cp; self.core = core
    def forward(self, past_hidden, code0_emb, rope_cos, rope_sin, slot_mask0, slot_mask1, attn_mask):
        x = self.cp.small_to_mtp_projection(torch.cat([past_hidden, code0_emb], dim=1))
        hs_ = []
        h = x
        for i in range(NL):
            layer = self.core.layers[i]; residual = h
            hn = layer.input_layernorm(h); at = layer.self_attn
            sh = hn.shape[:-1]; shp = (*sh,-1,HD)
            q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
            k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
            v = at.v_proj(hn).view(shp).transpose(1,2)
            qr = q*rope_cos + rh(q)*rope_sin
            kr = k*rope_cos + rh(k)*rope_sin
            zv = torch.zeros(1,NKV,CAP,HD)
            kvk = zv*(1.0-slot_mask0-slot_mask1) + kr[:,:,0:1,:]*slot_mask0 + kr[:,:,1:2,:]*slot_mask1
            kvv = zv*(1.0-slot_mask0-slot_mask1) + v[:,:,0:1,:]*slot_mask0 + v[:,:,1:2,:]*slot_mask1
            g = NH//NKV
            sc = torch.matmul(qr, kvk.repeat_interleave(g,dim=1).transpose(-1,-2))*SCALE
            pr = torch.softmax(sc + attn_mask, dim=-1)
            ao = torch.matmul(pr, kvv.repeat_interleave(g,dim=1)).reshape(*sh,-1).contiguous()
            h2 = residual + at.o_proj(ao); r2 = h2
            h = r2 + layer.mlp(layer.post_attention_layernorm(h2))
            hs_.append(h)
        return tuple(hs_) + (self.core.norm(h),)
m = P5(cp, core).eval()
tew = np.fromfile(CR + '/talker_codec_emb_weight.f32', dtype=np.float32).reshape(3072,2048)
past = np.fromfile(D + '/vd_m5b/prefill_hidden_last.raw', dtype=np.float32).reshape(1,1,2048)
c0e = tew[1995].reshape(1,1,2048)
x0 = torch.zeros(1,1,1024); pos = torch.arange(CAP).unsqueeze(0)
with torch.no_grad(): c, si = core.rotary_emb(x0, pos)
c = np.asarray(c, np.float32); si = np.asarray(si, np.float32)
sm0 = np.zeros((1,1,CAP,1), np.float32); sm0[0,0,0,0]=1
sm1 = np.zeros((1,1,CAP,1), np.float32); sm1[0,0,1,0]=1
am = np.full((1,1,2,CAP), -1e4, np.float32); am[0,0,0,0]=0; am[0,0,1,0]=0; am[0,0,1,1]=0
ins = {'past_hidden': past, 'code0_emb': c0e,
       'rope_cos': c[:,:2].reshape(1,1,2,HD), 'rope_sin': si[:,:2].reshape(1,1,2,HD),
       'slot_mask0': sm0, 'slot_mask1': sm1, 'attn_mask': am}
names = ['h%d' % i for i in range(NL)] + ['out']
torch.onnx.export(m, (torch.from_numpy(past), torch.from_numpy(c0e),
                      torch.from_numpy(ins['rope_cos']), torch.from_numpy(ins['rope_sin']),
                      torch.from_numpy(sm0), torch.from_numpy(sm1), torch.from_numpy(am)),
                  CR + '/dbg_p5.onnx', input_names=list(ins.keys()), output_names=names,
                  opset_version=18, dynamo=False)
so = ort.InferenceSession(CR + '/dbg_p5.onnx', providers=['CPUExecutionProvider'])
ox = dict(zip(names, so.run(None, ins)))
# torch reference per layer using the model's own rope fn
with torch.no_grad():
    xp = cp.small_to_mtp_projection(torch.cat([torch.from_numpy(past), torch.from_numpy(c0e)], dim=1))
    h = xp
    for i in range(NL):
        layer = core.layers[i]; residual = h
        hn = layer.input_layernorm(h); at = layer.self_attn
        sh = hn.shape[:-1]; shp = (*sh,-1,HD)
        q = at.q_norm(at.q_proj(hn).view(shp)).transpose(1,2)
        k = at.k_norm(at.k_proj(hn).view(shp)).transpose(1,2)
        v = at.v_proj(hn).view(shp).transpose(1,2)
        qr, kr = MF.apply_rotary_pos_emb(q, k, torch.from_numpy(c[:,:2]), torch.from_numpy(si[:,:2]))
        tmp = torch.zeros(1,NKV,CAP,HD)
        kvk = tmp*(1.0-torch.from_numpy(sm0)-torch.from_numpy(sm1)) + kr[:,:,0:1,:]*torch.from_numpy(sm0) + kr[:,:,1:2,:]*torch.from_numpy(sm1)
        kvv = tmp*(1.0-torch.from_numpy(sm0)-torch.from_numpy(sm1)) + v[:,:,0:1,:]*torch.from_numpy(sm0) + v[:,:,1:2,:]*torch.from_numpy(sm1)
        sc = torch.matmul(qr, kvk.repeat_interleave(2,dim=1).transpose(-1,-2))*SCALE
        pr = torch.softmax(sc + torch.from_numpy(am), dim=-1)
        ao = torch.matmul(pr, kvv.repeat_interleave(2,dim=1)).reshape(*sh,-1).contiguous()
        h2 = residual + at.o_proj(ao); r2 = h2
        h = r2 + layer.mlp(layer.post_attention_layernorm(h2))
        print('layer %d  cos(torch h, onnx h) = %.7f' % (i, cos(h.numpy(), ox['h%d'%i])))
    hf = core.norm(h)
    print('final     cos(torch norm(h), onnx out) = %.7f' % cos(hf.numpy(), ox['out']))
    o = core(inputs_embeds=xp, use_cache=True)
    print('model     cos(torch core(), onnx out)  = %.7f' % cos(o.last_hidden_state.numpy(), ox['out']))
print('DBG5_DONE')
