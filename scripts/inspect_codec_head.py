import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
ch = talker.codec_head
print('codec_head:', type(ch).__name__, tuple(ch.weight.shape))
print('bias:', hasattr(ch, 'bias') and ch.bias is not None)
print('vocab:', talker.config.vocab_size, 'hidden:', talker.config.hidden_size)
# 是否 tie weights
print('codec_embedding weight shape:', tuple(talker.get_input_embeddings().weight.shape))
same = torch.equal(ch.weight, talker.get_input_embeddings().weight)
print('tied to embedding:', same)
print('DONE', flush=True)