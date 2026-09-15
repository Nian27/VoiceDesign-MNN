import wave, numpy as np, os, shutil
SRC = r'E:/AndroidStudioProjects/qwen3tts-mnn/artifacts/device-smoke/app_wav/reference_opencl_device.wav'
w = wave.open(SRC)
ch, sr, n, sw = w.getnchannels(), w.getframerate(), w.getnframes(), w.getsampwidth()
d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32768.0
print('channels=%d rate=%d frames=%d sampwidth=%d dur=%.3fs' % (ch, sr, n, sw, n/sr))
print('peak=%.4f rms=%.5f zero_frac=%.4f dc=%.5f' % (float(np.abs(d).max()), float(np.sqrt((d**2).mean())),
      float((np.abs(d)<1e-4).mean()), float(d.mean())))
# 每 0.5s 的 RMS，看是否有内容/静音段
seg = int(0.5*sr)
rms = [float(np.sqrt((d[i:i+seg]**2).mean())) for i in range(0, len(d)-seg, seg)]
print('每 0.5s RMS:', ' '.join('%.3f' % r for r in rms[:17]))
print('non-silent segments(>0.01): %d/%d' % (sum(1 for r in rms if r>0.01), len(rms)))
DST = r'E:/AndroidStudioProjects/ReaderVoiceMobile/output/voicedesign_reference_opencl.wav'
os.makedirs(os.path.dirname(DST), exist_ok=True)
shutil.copy2(SRC, DST)
print('COPY_TO:', DST)
