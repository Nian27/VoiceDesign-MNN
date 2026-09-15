import io
p = r'E:/AndroidStudioProjects/qwen3tts-mnn/scripts/export/vd_cpr1_export2.py'
s = io.open(p, encoding='utf-8').read()
# 1) 让两个类存 submodule
s = s.replace('class Prefill2(torch.nn.Module):\n    def forward(self',
              'class Prefill2(torch.nn.Module):\n    def __init__(self, cp, core):\n        super().__init__(); self.cp = cp; self.core = core\n    def forward(self')
s = s.replace('class Step2(torch.nn.Module):\n    def forward(self',
              'class Step2(torch.nn.Module):\n    def __init__(self, cp, core):\n        super().__init__(); self.cp = cp; self.core = core\n    def forward(self')
# 2) 类体内引用改成 self.
n1 = s.count('x = cp.small_to_mtp_projection')
s = s.replace('x = cp.small_to_mtp_projection', 'x = self.cp.small_to_mtp_projection')
n2 = s.count('core.layers[i]')
s = s.replace('core.layers[i]', 'self.core.layers[i]')
n3 = s.count('core.norm(h)')
s = s.replace('core.norm(h)', 'self.core.norm(h)')
# 3) 实例化传参
s = s.replace('pref = Prefill2().eval(); stepm = Step2().eval()', 'pref = Prefill2(cp, core).eval(); stepm = Step2(cp, core).eval()')
n4 = s.count('Prefill2(cp, core)')
io.open(p, 'w', encoding='utf-8').write(s)
print('init-added=%d/%d  proj=%d layers=%d norm=%d inst=%d' % (s.count('self.cp = cp'), s.count('self.core = core'), n1, n2, n3, n4))