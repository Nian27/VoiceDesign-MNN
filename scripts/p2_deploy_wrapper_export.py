# P2: GraphBDeployWrapper -> fixed-shape ONNX (MAX_KV static, explicit mask)
import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
MAX_KV = 768
KV_IN = 767
print('loading fp32...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
from transformers.cache_utils import DynamicCache
class GraphBDeployWrapper(torch.nn.Module):
    def __init__(self, base, max_kv):
        super().__init__()
        self.base = base
        self.max_kv = max_kv
    def forward(self, inputs_embeds, position_ids, cache_position, kv_k, kv_v, attention_mask):
        # deploy: KV full MAX_KV, explicit mask input [1,1,1,MAX_KV] (0 visible / -inf masked)
        kv = DynamicCache()
        for i in range(28):
            kv.update(kv_k[i].contiguous(), kv_v[i].contiguous(), i)
        out = self.base(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=kv,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        return out.last_hidden_state, out.hidden_states[-1]
core = GraphBDeployWrapper(base, MAX_KV).eval()
B,H,D,Cd = 1, 8, 128, 2048
emb = torch.zeros((1,1,Cd), dtype=torch.float32)
pos = torch.zeros((3,1,1), dtype=torch.float32)
pos[0,0,0]=62; pos[1,0,0]=62; pos[2,0,0]=62
cp_ = torch.tensor([62], dtype=torch.int64)
kk = torch.zeros((28,B,H,KV_IN,D), dtype=torch.float32)
vv = torch.zeros((28,B,H,KV_IN,D), dtype=torch.float32)
amask = torch.zeros((1,1,1,MAX_KV), dtype=torch.float32)
amask[0,0,0,63:] = float('-inf')  # 768 mask: query at 62 sees 0..62
amask[0,0,0,62:63] = 0.0  # query at 62 sees 0..62
with torch.inference_mode():
    y1, y2 = core(emb, pos, cp_, kk, vv, amask)
print('fwd ok', tuple(y1.shape), flush=True)
torch.onnx.export(
    core,
    (emb, pos, cp_, kk, vv, amask),
    ART + '/graphb_deploy_m768.onnx',
    input_names=['inputs_embeds','position_ids','cache_position','kv_k','kv_v','attention_mask'],
    output_names=['hidden_states','layer_hidden'],
    opset_version=18,
    dynamo=False,
)
print('GRAPHB_DEPLOY_M768_EXPORT_OK', flush=True)