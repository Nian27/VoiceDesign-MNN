import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
L = open(r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/eos.log', 'w')
def log(s):
    L.write(s + '\n'); L.flush()
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
tc = gw.model.talker.config
for k in ['codec_eos_token_id','codec_bos_token_id','codec_pad_id','codec_pad_token_id','codec_nothink_id','codec_think_id','codec_think_bos_id','codec_think_eos_id','num_code_groups','vocab_size','spk_id','codec_language_id']:
    log(k + '=' + str(getattr(tc, k, 'MISSING')))
log('DONE')