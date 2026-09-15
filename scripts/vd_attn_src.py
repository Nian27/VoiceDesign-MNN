import inspect, os
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers.models.qwen3 import modeling_qwen3 as Q3
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
gwfn = ALL_ATTENTION_FUNCTIONS.get('sdpa')
print('sdpa interface =', gwfn)
try:
    src = inspect.getsource(gwfn)
    print(src[:2600])
except Exception as e:
    print('src err', e)
print('=== repeat_kv ===')
try:
    print(inspect.getsource(Q3.repeat_kv)[:700])
except Exception as e:
    print('rk err', e)
