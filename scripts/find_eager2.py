import transformers, inspect
from transformers.integrations.flash_attention import eager_attention_forward as ef
print('module:', ef.__module__)
print('sig:', inspect.signature(ef))
src = inspect.getsource(ef)
print(src[:2600])