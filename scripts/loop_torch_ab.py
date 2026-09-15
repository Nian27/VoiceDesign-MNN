# Gate4 torch authoritative: GraphA prefill -> GraphB decode 20 steps (KV 35 contract)
import os, json, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
G = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/m22/golden'
NUM_STEPS = 20
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cpm = talker.code_predictor.model
lm = talker.code_predictor.lm_head
cp_emb = talker.code_predictor.get_input_embeddings()
proj = talker.code_predictor.small_to_mtp_projection
LOG = open(D + '/loop_torch_ab_diag.log', 'w')
def log(s):
    LOG.write(s + '\n'); LOG.flush()
import os as _os
_os.makedirs(D + '/loop_ab_pt', exist_ok=True)
from transformers.cache_utils import DynamicCache
log('LOADED')
# GraphA prefill: use golden prefill emb (34 tokens) to build KV 35
# golden has prefill? decode_steps only. Build prefill from GraphA inputs in m22_golden.json
# use decode emb (1,1,2048) at cp=34 as FIRST token appended to 34-token prefill KV
# Simplest authoritative: run prefill 34 tokens with known emb (from m22_golden.json event0: inputs_embeds (1,34,2048))
import json as _json
mj = _json.load(open(G + '/m22_golden.json'))
ev0 = mj['events']['decode'][0]
print('prefill event shapes:', ev0['inputs_embeds']['shape'], ev0['position_ids']['shape'], flush=True)
log('prefill shapes ' + str(ev0['inputs_embeds']['shape']))
# We need actual prefill emb tensor - not in golden json (samples only). Reconstruct: use golden step0 emb as decode first token
# and prefill KV via talker.model over 34-token inputs_embeds generated from golden? golden lacks prefill tensor.
# Fallback: build prefill emb as golden step0 emb repeated? NOT authoritative.
log('PREFILL_TENSOR_NOT_IN_GOLDEN - need capture')
print('NEED_PREFILL_CAPTURE', flush=True)