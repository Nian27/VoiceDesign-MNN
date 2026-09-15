import os, torch, json, time
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART + '/onnx', exist_ok=True)
from qwen_tts import Qwen3TTSModel
print('loading...', flush=True)
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
m = gw.model; talker = m.talker; base = talker.model

# Graph B per M2.2 spec: talker base single decode step, talker domain 2048,
# static KV 28x2 (1,8,T,128), mrope position_ids (3,1,1), CP runs host-side (G1 captured).
class TalkerBaseStep(torch.nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.talker = talker
    def forward(self, inputs_embeds, position_ids, kv_k, kv_v, cache_position):
        from transformers.cache_utils import DynamicCache
        kv = DynamicCache()
        for i in range(len(kv_k)):
            kv.update(kv_k[i], kv_v[i], i)
        out = self.talker.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=kv,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=False,
        )
        logits = self.talker.codec_head(out.last_hidden_state)
        return out.last_hidden_state, logits

core = TalkerBaseStep(talker).eval()
B, H, T, D, d = 1, 8, 34, 128, 2048
dev = base.device
emb = torch.randn((1,1,D), dtype=torch.float32, device=dev)
pos = torch.zeros((3,1,1), dtype=torch.float32, device=dev)
pos[0,0,0]=34; pos[1,0,0]=34; pos[2,0,0]=34
kk = [torch.zeros((1,H,T,d), dtype=torch.float32, device=dev) for _ in range(28)]
vv = [torch.zeros((1,H,T,d), dtype=torch.float32, device=dev) for _ in range(28)]
cp_ = torch.tensor([34], dtype=torch.int64, device=dev)
print('dummy created', flush=True)
for _ in range(3):
    y1, y2 = core(emb, pos, kk, vv, cp_)
print('fwd ok', tuple(y1.shape), tuple(y2.shape), flush=True)
try:
    torch.onnx.export(
        core,
        (emb, pos, kk, vv, cp_),
        ART + '/onnx/graphb_talker_base_step.onnx',
        input_names=['inputs_embeds','position_ids','kv_k','kv_v','cache_position'],
        output_names=['hidden_states','logits'],
        dynamic_axes={'inputs_embeds': {0:'batch'}, 'position_ids': {0:'mrope'}, 'kv_k': {0:'batch',2:'kv_len'}, 'kv_v': {0:'batch',2:'kv_len'}},
        opset_version=18,
        dynamo=False,
    )
    print('GRAPH_B_EXPORT_OK', flush=True)
except Exception as e:
    print('GRAPH_B_EXPORT_FAIL', repr(e)[:700], flush=True)
print('DONE', flush=True)