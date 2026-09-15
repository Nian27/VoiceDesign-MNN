import numpy as np, wave, struct, json
D = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke'
dev = np.fromfile(D + '/decoder_pcm_dev_cpu_f32.raw', dtype=np.float32)
pc = np.fromfile(D + '/decoder_pcm_pc_f32.raw', dtype=np.float32)
n = min(len(dev), len(pc)); a = dev[:n].astype(np.float64); b = pc[:n].astype(np.float64)
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
mae = float(np.mean(np.abs(a-b))); mx = float(np.max(np.abs(a-b)))
rel_l2 = float(np.linalg.norm(a-b)/np.linalg.norm(a))
si = 10*np.log10(np.sum(a*a)/(np.sum((a-b)**2)+1e-9))
res = {'n': int(n), 'cosine': round(cos,6), 'mae': round(mae,8), 'maxabs': round(mx,6), 'rel_l2': round(rel_l2,6), 'si_sdr_db': round(float(si),2)}
# Audio gate: wav @24kHz from DEVICE pcm
iv = np.clip(dev, -1, 1)
pcm16 = (iv * 32767).astype('<i2')
with wave.open(D + '/decoder_device_smoke.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm16.tobytes())
print(json.dumps(res))
print('WAV written: decoder_device_smoke.wav', len(pcm16), 'samples @24000Hz =', round(len(pcm16)/24000,2), 's')