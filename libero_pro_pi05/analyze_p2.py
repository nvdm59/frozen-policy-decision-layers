import json,glob,collections
from math import comb
R=collections.defaultdict(dict)
for f in glob.glob('results/p2_heldout/*.jsonl'):
    for l in open(f):
        r=json.loads(l); R[r['arm']][(r['suite'],r['task_id'],r['trial'])]=r
B,Lr=R['base'],R['layer']
print('n base',len(B),'layer',len(Lr))
def mcn(keys):
    w=sum(1 for k in keys if Lr[k]['success'] and not B[k]['success']); l=sum(1 for k in keys if B[k]['success'] and not Lr[k]['success'])
    n=w+l; p=min(1,2*sum(comb(n,i) for i in range(0,min(w,l)+1))/2**n) if n else 1
    return w,l,p
allk=sorted(set(B)&set(Lr))
for s in sorted({k[0] for k in allk})+['ALL']:
    ks=[k for k in allk if s=='ALL' or k[0]==s]
    print(s,'base',sum(B[k]['success'] for k in ks),'layer',sum(Lr[k]['success'] for k in ks),'/',len(ks),'win/loss/p',mcn(ks))
for s in sorted({k[0] for k in allk}):
    for t in range(10):
        ks=[k for k in allk if k[0]==s and k[1]==t]
        if ks: print(' ',s[7:-5],t,'base',sum(B[k]['success'] for k in ks),'layer',sum(Lr[k]['success'] for k in ks),'/',len(ks),'carried',sum(any(e[0]=='carry' for e in Lr[k]['events']) for k in ks),Lr[ks[0]]['target'],'->',Lr[ks[0]]['dest'],Lr[ks[0]]['why'],'|',Lr[ks[0]]['prompt'][:70])
