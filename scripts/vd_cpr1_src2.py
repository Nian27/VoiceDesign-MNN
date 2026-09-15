import os, inspect, re
os.environ['HF_HUB_OFFLINE'] = '1'
import qwen_tts.core.models.modeling_qwen3_tts as M
src = inspect.getsource(M)
i = src.find('class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration')
j = src.find('\nclass ', i + 10)
body = src[i:j if j > 0 else i+9000]
# real forward (exact match, not forward_finetune)
for m in re.finditer(r'\n    def (\w+)\(', body):
    print('method:', m.group(1))
print()
k = body.find('\n    def forward(')
if k < 0:
    print('NO plain forward')
else:
    print('===== plain forward =====')
    print(body[k:k+2400])
