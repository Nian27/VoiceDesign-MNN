import os, torch, json, time
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART + '/onnx', exist_ok=True)
print('loading...', flush=True)
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
m = gw.model; talker = m.talker; base = talker.model
from transformers.cache_utils import DynamicCache
class ReplayStep(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, inputs_embeds, position_ids, kv_k, kv_v, cache_position):
        kv = DynamicCache()
        for i in range(kv_k.shape[0]):
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
core = ReplayStep(base).eval()
B,H,T,D,Cd = 1, 8, 34, 128, 2048
dev = next(base.parameters()).device
emb = torch.zeros((1,1,Cd), dtype=torch.bfloat16, device=dev)
pos = torch.zeros((3,1,1), dtype=torch.float32, device=dev)
pos[0,0,0]=34; pos[1,0,0]=34; pos[2,0,0]=34
kk = torch.zeros((28,1,H,T,D), dtype=torch.bfloat16, device=dev)
vv = torch.zeros((28,1,H,T,D), dtype=torch.bfloat16, device=dev)
cp_ = torch.tensor([34], dtype=torch.int64, device=dev)
print('dummy created', flush=True)
y1, y2 = core(emb, pos, kk, vv, cp_)
print('fwd ok', tuple(y1.shape), tuple(y2.shape), flush=True)
try:
    torch.onnx.export(
        core,
        (emb, pos, kk, vv, cp_),
        ART + '/onnx/graphb_replay_step.onnx',
        input_names=['inputs_embeds','position_ids','kv_k','kv_v','cache_position'],
        output_names=['hidden_states','layer_hidden'],
        dynamic_shapes={'inputs_embeds': {0: torch.export.Dim('batch')}, 'position_ids': {0: torch.export.Dim('mrope')}, 'kv_k': {0: torch.export.Dim('layers'), 1: torch.export.Dim('batch'), 3: torch.export.Dim('kv_len')}, 'kv_v': {0: torch.export.Dim('layers'), 1: torch.export.Dim('batch'), 3: torch.export.Dim('kv_len')}, 'cache_position': None},
        opset_version=18,
        dynamo=True,
    )
    print('GRAPH_B_REPLAY_EXPORT_OK', flush=True)
except Exception as e:
    print('GRAPH_B_REPLAY_EXPORT_FAIL', repr(e)[:700], flush=True)
print('DONE', flush=True)