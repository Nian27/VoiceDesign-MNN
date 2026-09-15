import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
os.makedirs(ART, exist_ok=True)
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
from transformers.cache_utils import DynamicCache
class ReplayStepStatic(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, inputs_embeds, position_ids, kv_k, kv_v, cache_position):
        kv = DynamicCache()
        for i in range(28):
            kv.update(kv_k[i], kv_v[i], i)
        out = self.base(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=kv,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
        )
        return out.last_hidden_state, out.hidden_states[-1]
core = ReplayStepStatic(base).eval()
B,H,T,D,Cd = 1, 8, 128, 128, 2048
emb = torch.zeros((1,1,Cd), dtype=torch.float32)
pos = torch.zeros((3,1,1), dtype=torch.float32)
pos[0,0,0]=34; pos[1,0,0]=34; pos[2,0,0]=34
kk = torch.zeros((28,B,H,T,D), dtype=torch.float32)
vv = torch.zeros((28,B,H,T,D), dtype=torch.float32)
cp_ = torch.tensor([34], dtype=torch.int64)
with torch.inference_mode():
    y1, y2 = core(emb, pos, kk, vv, cp_)
print('fwd ok', tuple(y1.shape), flush=True)
torch.onnx.export(
    core,
    (emb, pos, kk, vv, cp_),
    ART + '/graphb_static_t128.onnx',
    input_names=['inputs_embeds','position_ids','kv_k','kv_v','cache_position'],
    output_names=['hidden_states','layer_hidden'],
    opset_version=18,
    dynamo=True,
)
print('GRAPHB_STATIC_T128_EXPORT_OK', flush=True)