import os, torch, json
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'E:/AndroidStudioProjects/qwen3tts-mnn/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
ART = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
from qwen_tts import Qwen3TTSModel
gw = Qwen3TTSModel.from_pretrained(MODEL, dtype=torch.float32, device_map='cpu')
talker = gw.model.talker
print('loaded', flush=True)
emb = talker.get_input_embeddings()
cp_emb = talker.code_predictor.get_input_embeddings()
print('emb types:', type(emb).__name__, 'cp_emb len:', len(cp_emb) if hasattr(cp_emb, '__len__') else 'n/a', flush=True)
code0 = torch.tensor([[2149]], dtype=torch.long)
last_hid = emb(code0)
print('last_hid', tuple(last_hid.shape), 'maxabs', round(float(last_hid.abs().max()),4), flush=True)
seqs = torch.randint(0, 2048, (1, 15))
codec_h = [last_hid] + [cp_emb[i](seqs[:, i:i+1]) for i in range(15)]
codec_sum = torch.cat(codec_h, dim=1).sum(1, keepdim=True)
print('codec_sum', tuple(codec_sum.shape), 'maxabs', round(float(codec_sum.abs().max()),4), flush=True)
tts_pad = talker.text_projection(emb(torch.tensor([[151671]], dtype=torch.long)))
print('tts_pad', tuple(tts_pad.shape), 'maxabs', round(float(tts_pad.abs().max()),4), flush=True)
inputs_embeds = codec_sum + tts_pad
print('frame_emb', tuple(inputs_embeds.shape), 'maxabs', round(float(inputs_embeds.abs().max()),4), flush=True)
torch.save({'last_hid': last_hid, 'seqs15': seqs, 'codec_sum': codec_sum, 'tts_pad': tts_pad, 'frame_emb': inputs_embeds}, ART + '/loop_frame_golden.pt')
print('LOOP_FRAME_SAVED', flush=True)