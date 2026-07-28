import importlib.util, sys, time
import numpy as np

BASE = {"n_sites":12,"n_layers":5,"beta":0.20,"single_ion_anisotropy":0.15,
        "learning_rate":0.03,"initial_parameter_scale":0.05,"seed":2041}

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m
    spec.loader.exec_module(m); return m

path=sys.argv[1]; name=sys.argv[2]
m=load(path,name)
t={}
for n in (20,120):
    cfg=dict(BASE, max_steps=n)
    t0=time.monotonic(); m.run_solution(cfg); t[n]=time.monotonic()-t0
    print(f"{name}: {n} steps -> {t[n]:.2f}s")
per=(t[120]-t[20])/100.0
fixed=t[20]-20*per
print(f"{name}: per-step {per*1000:.1f}ms, fixed(compile+final) {fixed:.1f}s, projected 500 = {fixed+500*per:.1f}s")
