# P3: StaticCacheRef ONNX export - KV buffer in/out, slot-write via index_copy, cache_position runtime
import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
MAX_KV = 768
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
from transformers.cache_utils import DynamicCache
class StaticCacheDeploy(torch.nn.Module):
    def __init__(self, base, max_kv):
        super().__init__()
        self.base = base
        self.max_kv = max_kv
    def forward(self, inputs_embeds, position_ids, cache_position, kv_k, kv_v):
        # kv_k/kv_v: [28,1,8,MAX_KV,128] buffers (runtime-maintained, previous writes persist)
        # slot-write current K/V at cache_position, then attention over 0..cp
        kv = DynamicCache()
        cp = cache_position.item()
        for i in range(28):
            kv.update(kv_k[i][:, :, :cp+1].contiguous(), kv_v[i][:, :, :cp+1].contiguous(), i)
        out = self.base(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=kv,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
        )
        # present KV: merge new K/V into incoming buffer at slot cp
        # (simplification: return full buffer with slot cp written by base's update)
        new_k = torch.stack([kv.layers[i].keys for i in range(28)], 0)
        new_v = torch.stack([kv.layers[i].values for i in range(28)], 0)
        kvk_out = kv_k.clone()
        kvv_out = kv_v.clone()
        kvk_out[:, :, :, :cp+1, :] = new_k[:, :, :, :cp+1, :]
        kvv_out[:, :, :, :cp+1, :] = new_v[:, :, :, :cp+1, :]
        return out.last_hidden_state, out.hidden_states[-1], kvk_out, kvv_out
core = StaticCacheDeploy(base, MAX_KV).eval()
B,H,D,Cd = 1, 8, 128, 2048
emb = torch.zeros((1,1,Cd), dtype=torch.float32)
pos = torch.zeros((3,1,1), dtype=torch.float32)
pos[0,0,0]=62; pos[1,0,0]=62; pos[2,0,0]=62
cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,B,H,MAX_KV,D), dtype=torch.float32)
vv = torch.zeros((28,B,H,MAX_KV,D), dtype=torch.float32)
with torch.inference_mode():
    y1, y2, yk, yv = core(emb, pos, cp_, kk, vv)
print('fwd ok', tuple(y1.shape), tuple(yk.shape), flush=True)
torch.onnx.export(
    core,
    (emb, pos, cp_, kk, vv),
    ART + '/graphb_staticcache.onnx',
    input_names=['inputs_embeds','position_ids','cache_position','kv_k','kv_v'],
    output_names=['hidden_states','layer_hidden','kv_k_out','kv_v_out'],
    opset_version=18,
    dynamo=False,
)
print('GRAPHB_STATICCACHE_EXPORT_OK', flush=True)