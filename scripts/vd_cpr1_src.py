import os, inspect
os.environ['HF_HUB_OFFLINE'] = '1'
import qwen_tts.core.models.modeling_qwen3_tts as M
src = inspect.getsource(M)
import re
# find the CodePredictor ForConditionalGeneration class and its forward
for cls in ['Qwen3TTSTalkerCodePredictorModelForConditionalGeneration','Qwen3TTSTalkerCodePredictorModel']:
    i = src.find('class ' + cls)
    if i < 0:
        print('NOT FOUND', cls); continue
    j = src.find('\nclass ', i + 10)
    body = src[i:j if j > 0 else i+9000]
    print('=' * 20, cls, 'len', len(body))
    # print forward + prepare_inputs
    for meth in ['def forward', 'def prepare_inputs_for_generation', 'def get_input_embeddings']:
        k = body.find(meth)
        if k >= 0:
            seg = body[k:k+1600]
            # cut at next def at same indent
            print('---', meth, '---')
            print(seg[:1500])
            print()
