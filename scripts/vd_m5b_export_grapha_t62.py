import os, torch, sys
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
LP = 62
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
base = gw.model.talker.model
base.config._attn_implementation = 'eager'
from transformers.cache_utils import DynamicCache
class PrefillStatic(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, inputs_embeds, position_ids):
        kv = DynamicCache()
        out = self.base(inputs_embeds=inputs_embeds, position_ids=position_ids,
                        past_key_values=kv, cache_position=torch.arange(LP),
                        use_cache=True, output_hidden_states=False)
        kk = torch.stack([kv.layers[i].keys for i in range(28)], dim=0)
        vv = torch.stack([kv.layers[i].values for i in range(28)], dim=0)
        return out.last_hidden_state, kk, vv
core = PrefillStatic(base).eval()
emb = torch.randn((1, LP, 2048))
pos = torch.zeros((3, 1, LP))
for j in range(LP):
    pos[0,0,j]=j; pos[1,0,j]=j; pos[2,0,j]=j
with torch.inference_mode():
    h, kk, vv = core(emb, pos)
print('fwd ok', tuple(h.shape), tuple(kk.shape))
torch.onnx.export(core, (emb, pos), ART + '/grapha_prefill_t62.onnx',
                  input_names=['inputs_embeds','position_ids'],
                  output_names=['hidden_states','kv_k','kv_v'],
                  opset_version=18, dynamo=False)
import onnx
mm = onnx.load(ART + '/grapha_prefill_t62.onnx', load_external_data=False)
ops = {}
for n in mm.graph.node: ops[n.op_type] = ops.get(n.op_type,0)+1
print('IsNaN =', ops.get('IsNaN',0), 'nodes', len(mm.graph.node))
print('size', os.path.getsize(ART + '/grapha_prefill_t62.onnx'))
print('GRAPHA_T62_EXPORT_DONE')
