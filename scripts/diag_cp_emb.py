import os, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign', dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
cp_emb = talker.code_predictor.get_input_embeddings()
print('cp_emb type:', type(cp_emb).__name__, 'len:', len(cp_emb))
e0 = cp_emb[0](torch.tensor([[5]]))
print('cp_emb[0] out:', tuple(e0.shape))
print('codepred config hidden:', talker.code_predictor.config.hidden_size)
print('talker codec_embedding:', talker.get_input_embeddings())
ce = talker.get_input_embeddings()(torch.tensor([[5]]))
print('talker codec_emb out:', tuple(ce.shape))
print('DONE', flush=True)