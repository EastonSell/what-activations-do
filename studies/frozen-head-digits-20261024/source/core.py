"""Float64 kernels; no real dataset imports and no experiment execution on import."""
import hashlib
import time
import numpy as np

KEYS=('W1','b1','W2','b2')

def rng(*seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))

def initialize(seed,inputs,hidden_dim,classes):
    gen=rng(seed,0); p={}
    for suffix,fan_in,fan_out in [('1',inputs,hidden_dim),('2',hidden_dim,classes)]:
        bound=np.sqrt(6/(fan_in+fan_out))
        p['W'+suffix]=gen.uniform(-bound,bound,(fan_in,fan_out))
        p['b'+suffix]=gen.uniform(-bound,bound,fan_out)
    return p

def matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)==0 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite matrix')
    return x

def hidden(p,x):
    x=matrix(x)
    if x.shape[1]!=p['W1'].shape[0]: raise ValueError('Input width mismatch')
    return np.maximum(x@p['W1']+p['b1'],0)

def head(p,h):
    return matrix(h)@p['W2']+p['b2']

def forward(p,x):
    return head(p,hidden(p,x))

def metrics(logits,y):
    logits=matrix(logits); y=np.asarray(y)
    if y.shape!=(len(logits),) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=logits.shape[1]):
        raise ValueError('Invalid integer class labels')
    z=logits-logits.max(axis=1,keepdims=True)
    logden=np.log(np.exp(z).sum(axis=1))
    correct_logit=logits[np.arange(len(y)),y]
    foil=logits.copy(); foil[np.arange(len(y)),y]=-np.inf
    return {'ce':logden-z[np.arange(len(y)),y],
            'margin':correct_logit-foil.max(axis=1),
            'correct':logits.argmax(axis=1)==y}

def loss_grad(p,x,y,alpha,n_train):
    x=matrix(x); h=hidden(p,x); z=head(p,h)
    m=metrics(z,y)
    if alpha<0 or n_train<len(x): raise ValueError('Invalid regularization scale')
    probs=np.exp(z-z.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
    probs[np.arange(len(y)),y]-=1; probs/=len(y)
    dh=(probs@p['W2'].T)*(h>0)
    g={'W1':x.T@dh+alpha/n_train*p['W1'], 'b1':dh.sum(axis=0),
       'W2':h.T@probs+alpha/n_train*p['W2'], 'b2':probs.sum(axis=0)}
    loss=m['ce'].mean()+alpha/(2*n_train)*(np.sum(p['W1']**2)+np.sum(p['W2']**2))
    return float(loss),g

class Adam:
    def __init__(self,p,learning_rate,beta1,beta2,epsilon):
        self.lr=learning_rate; self.b1=beta1; self.b2=beta2; self.eps=epsilon; self.t=0
        self.m={k:np.zeros_like(v) for k,v in p.items()}; self.v={k:np.zeros_like(v) for k,v in p.items()}
    def step(self,p,g,keys):
        self.t+=1
        for key in keys:
            self.m[key]=self.b1*self.m[key]+(1-self.b1)*g[key]
            self.v[key]=self.b2*self.v[key]+(1-self.b2)*g[key]**2
            m=self.m[key]/(1-self.b1**self.t); v=self.v[key]/(1-self.b2**self.t)
            p[key]-=self.lr*m/(np.sqrt(v)+self.eps)
            if not np.isfinite(p[key]).all(): raise FloatingPointError('Nonfinite parameter')

def epoch_order(n,seed,epoch):
    return rng(seed,1,epoch).permutation(n)

def train(initial,x,y,seed,cfg,freeze_hidden=False,check=lambda:None):
    x=matrix(x); metrics(forward(initial,x),y)
    p={k:v.copy() for k,v in initial.items()}
    opt=Adam(p,cfg['learning_rate'],cfg['beta1'],cfg['beta2'],cfg['epsilon'])
    keys=('W2','b2') if freeze_hidden else KEYS
    history=[]; started=time.monotonic(); prior=started
    for epoch in range(cfg['epochs']):
        order=epoch_order(len(y),seed,epoch)
        for start in range(0,len(y),cfg['batch_size']):
            check(); ix=order[start:start+cfg['batch_size']]
            _,g=loss_grad(p,x[ix],y[ix],cfg['alpha'],len(y)); opt.step(p,g,keys)
        loss,_=loss_grad(p,x,y,cfg['alpha'],len(y))
        now=time.monotonic()
        history.append({'epoch':epoch+1,'loss':loss,'updates':opt.t,
                        'epoch_seconds':now-prior,'elapsed_seconds':now-started,
                        'order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()})
        prior=now
    if freeze_hidden:
        for key in ('W1','b1'):
            if p[key].tobytes()!=initial[key].tobytes(): raise AssertionError('Frozen hidden parameter changed')
    return p,history

def fit_geometry(h):
    h=matrix(h); mu=h.mean(axis=0)
    _,s,vt=np.linalg.svd(h-mu,full_matrices=False)
    if vt.shape!=(h.shape[1],h.shape[1]): raise ValueError('Training rows must cover hidden width')
    return mu,vt.T,s

def haar(gen,width):
    q,r=np.linalg.qr(gen.normal(size=(width,width)))
    signs=np.sign(np.diag(r)); signs[signs==0]=1
    return q*signs

def random_bases(seed,draw,width):
    return haar(rng(seed,2,draw,0),width),haar(rng(seed,2,draw,1),width)

def intervene(h,mu,basis,k,kind,rotation=None):
    h=matrix(h); width=h.shape[1]
    if not isinstance(k,(int,np.integer)) or not 0<=k<=width: raise ValueError('Invalid rank')
    if np.shape(mu)!=(width,) or np.shape(basis)!=(width,width): raise ValueError('Geometry shape mismatch')
    if kind not in ('pca','rank','energy'): raise ValueError('Unknown intervention')
    delta=((h-mu)@basis[:,:k])@basis[:,:k].T
    if kind=='energy':
        if np.shape(rotation)!=(width,width): raise ValueError('Missing rotation')
        delta=delta@rotation
    return h-delta

def gates(a,b,floor,gap):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=1 or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid gate arrays')
    primary=bool(np.all(a>=floor))
    return {'primary_pass':primary,'comparison_pass':bool(primary and np.all(b>=floor) and np.all(np.abs(a-b)<=gap))}

def bootstrap_cases(seed_by_case,draws,seed):
    a=matrix(seed_by_case); by_case=a.mean(axis=0); gen=rng(seed)
    values=np.empty(draws)
    for start in range(0,draws,100):
        count=min(100,draws-start)
        ix=gen.integers(0,len(by_case),size=(count,len(by_case)))
        values[start:start+count]=by_case[ix].mean(axis=1)
    return {'mean':float(by_case.mean()),'per_seed':a.mean(axis=1).tolist(),
            'ci95':np.quantile(values,[.025,.975]).tolist(),
            'scope':'Conditional on fixed models, reused test cases, and sampled controls'}
