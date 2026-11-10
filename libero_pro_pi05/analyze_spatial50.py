import json,glob
from math import comb
def load(d):
    R={}
    for f in glob.glob(f'results/{d}/*.jsonl'):
        for l in open(f):
            r=json.loads(l); R[(r['task_id'],r['trial'])]=r
    return R
B,L=load('p2_spatial_base50'),load('p2_spatial_layer50')
ks=sorted(set(B)&set(L)); print('base',len(B),'layer',len(L),'pairs',len(ks))
w=sum(L[k]['success'] and not B[k]['success'] for k in ks); l=sum(B[k]['success'] and not L[k]['success'] for k in ks)
n=w+l; p=min(1,2*sum(comb(n,i) for i in range(min(w,l)+1))/2**n)
print('base',sum(B[k]['success'] for k in ks),'layer',sum(L[k]['success'] for k in ks),'wins',w,'losses',l,'p',p)
for t in range(10):
    kt=[k for k in ks if k[0]==t]
    print(t,'base',sum(B[k]['success'] for k in kt),'layer',sum(L[k]['success'] for k in kt),'/',len(kt),'carried',sum(any(e[0]=='carry' for e in L[k]['events']) for k in kt),'|',L[kt[0]]['prompt'][18:62] if kt else '')
