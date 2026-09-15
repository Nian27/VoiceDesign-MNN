import torch, inspect
from transformers.modeling_attn_utils import ALL_ATTENTION_FUNCTIONS
fn = ALL_ATTENTION_FUNCTIONS['eager']
print('module:', fn.__module__)
print('sig:', inspect.signature(fn))
print(inspect.getsource(fn)[:2600])