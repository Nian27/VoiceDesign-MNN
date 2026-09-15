# VD-M5B: (a) compute prefill hidden+KV (torch, authoritative) for the real description
#         (b) export frame_emb graph: codes(1,16) -> next inputs_embeds(1,1,2048)
import os, numpy as np, torch, functools
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
OUT = D + '/vd_m5b'
os.makedirs(OUT, exist_ok=True)
from qwen_tts import Qwen3TTSModel
from transformers.cache_utils import DynamicCache
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
base = talker.model
cfg = gw.model.config
# --- (a) prefill capture + compute hidden/KV ---
cap = {}
orig = base.forward
@functools.wraps(orig)
def patched(*args, **kwargs):
    emb = kwargs.get('inputs_embeds', args[0] if args else None)
    if emb is not None and 'emb' not in cap:
        cap['emb'] = emb.detach().float().clone()
        cap['pos'] = kwargs.get('position_ids').detach().float().clone()
    return orig(*args, **kwargs)
base.forward = patched
texts = ['小家伙，修炼之事急不得，根基若是不稳，日后吃亏的还是你自己。']
instructs = ['苍老男性，低沉略沙哑，声音有岁月感，语气沉稳、有威严，但并不凶狠。']
try:
    gw.generate_voice_design(text=texts, instruct=instructs, language='Chinese', non_streaming_mode=True, max_new_tokens=1)
except Exception as e:
    pass
base.forward = orig
LP = int(cap['emb'].shape[1])
print('prefill len', LP)
kv = DynamicCache()
with torch.inference_mode():
    out = base(inputs_embeds=cap['emb'], position_ids=cap['pos'], past_key_values=kv,
               cache_position=torch.arange(LP), use_cache=True)
h = out.last_hidden_state.detach().float()
kk = torch.stack([kv.layers[i].keys for i in range(28)], dim=0).float()
vv = torch.stack([kv.layers[i].values for i in range(28)], dim=0).float()
print('hidden', tuple(h.shape), 'kv', tuple(kk.shape))
(h[:, -1:, :].contiguous().numpy()).tofile(OUT + '/prefill_hidden_last.raw')
kk.numpy().astype(np.float32).tofile(OUT + '/prefill_kv_k.raw')
vv.numpy().astype(np.float32).tofile(OUT + '/prefill_kv_v.raw')
np.save(OUT + '/prefill_cp.npy', np.array([LP], dtype=np.int32))
print('saved: hidden', os.path.getsize(OUT+'/prefill_hidden_last.raw'), 'kvk', os.path.getsize(OUT+'/prefill_kv_k.raw'))
# --- (b) frame_emb graph ---
codec_emb = talker.get_input_embeddings()
cp_emb = talker.code_predictor.get_input_embeddings()
pad_id = int(cfg.tts_pad_token_id)
with torch.no_grad():
    tts_pad = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[pad_id]])))  # (1,1,2048)
print('tts_pad_id', pad_id, 'pad shape', tuple(tts_pad.shape))
class FrameEmb(torch.nn.Module):
    def __init__(self, codec_emb, cp_emb, pad):
        super().__init__()
        self.codec_emb = codec_emb
        self.cp_emb = cp_emb
        self.register_buffer('pad', pad.clone())
    def forward(self, codes):           # (1,16) int64
        e = self.codec_emb(codes[:, :1])
        for g in range(15):
            e = e + self.cp_emb[g](codes[:, g+1:g+2])
        return e + self.pad
fe = FrameEmb(codec_emb, cp_emb, tts_pad).eval()
ct = torch.randint(0, 2048, (1, 16))
with torch.inference_mode():
    y = fe(ct)
print('frame_emb out', tuple(y.shape), 'maxabs %.3f' % float(y.abs().max()))
torch.onnx.export(fe, (ct,), OUT + '/frame_emb.onnx', input_names=['codes'], output_names=['emb'], opset_version=18, dynamo=False)
import onnx
mm = onnx.load(OUT + '/frame_emb.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('frame_emb IsNaN', ops.get('IsNaN',0), 'size', os.path.getsize(OUT + '/frame_emb.onnx'))
print('VD_M5B_PREP_DONE')
