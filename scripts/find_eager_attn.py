import inspect
from transformers import ALL_ATTENTION_FUNCTIONS
fn = ALL_ATTENTION_FUNCTIONS.get('eager')
print('eager fn:', fn)
src = inspect.getsource(fn)
print(src[:2500])