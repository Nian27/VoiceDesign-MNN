import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22'
os.makedirs(ART + '/onnx', exist_ok=True)
from qwen_tts import Qwen3TTSModel
print('loading...', flush=True)
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
m = gw.model; talker = m.talker; base = talker.model
layer0 = base.layers[0]
class LayerDecodeStep(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, hidden_states, position_ids, kv_k, kv_v, cache_position):
        kv_tuple = ((kv_k, kv_v),)
        out = self.layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=kv_tuple,
            cache_position=cache_position,
            use_cache=True,
            output_attentions=False,
        )
        return out[0], out[1][0], out[1][1]