"""Past-only targets and small frozen-state probes; no changes to the upstream encoder."""
import numpy as np
import torch
from torch import nn

PREFIXES = (32, 64, 96, 128)
HISTORY = 16
ALPHAS = (1e-4, 1e-3, 1e-2, .1)
METRICS = ('primary', 'path', 'body', 'activity', 'change1', 'close_mae_bps')


def targets(data, stats, prefix):
    """State h[prefix-1] reads rows [0,prefix); score16 rows strictly before it.

    Rebase the cached cumulative path at the close before this local history.
    Never slice the old normalized path without moving its price anchor.
    """
    if prefix not in PREFIXES:
        raise ValueError('Unregistered prefix')
    start, stop = prefix-HISTORY-1, prefix-1
    scale, mean = np.asarray(stats['y_scale']), np.asarray(stats['y_mean'])
    y = np.asarray(data['y'][:, start:stop], dtype=np.float64)*scale+mean
    anchor = np.asarray(data['y'][:, start-1, 0], dtype=np.float64)*scale[0]+mean[0]
    y[..., 0] -= anchor[:, None]
    mask = np.asarray(data['mask'][:, start:stop], dtype=bool).copy()
    if not mask[..., 0].all() or not np.asarray(data['mask'][:, start-1, 0]).all():
        raise ValueError('Local price targets require observed anchors and closes')
    return np.where(mask, y, 0).astype(np.float32), mask


def fit_target_scales(rows):
    """One shared channel scale, fitted to the same fixed train windows/all prefixes."""
    y = np.concatenate([v[0] for v in rows]); m = np.concatenate([v[1] for v in rows])
    count = m.sum((0,1)); mean = (y*m).sum((0,1),dtype=np.float64)/count.clip(1)
    scale = np.sqrt(((y-mean)**2*m).sum((0,1))/count.clip(1)).clip(1e-5)
    if (count<2).any():
        raise ValueError('Insufficient target support')
    delta = np.diff(y[...,0],axis=1)
    return dict(mean=mean.tolist(),scale=scale.tolist(),count=count.tolist(),
                delta_scale=max(float(np.sqrt(np.mean(delta.astype(float)**2))),.01),fitted_on='original_fixed_train_all_prefixes')


def normalize(y, mask, stats):
    return np.where(mask,(y-np.array(stats['mean']))/np.array(stats['scale']),0).astype(np.float32)


def feature_scales(x):
    return dict(mean=np.mean(x,axis=0,dtype=np.float64).tolist(),scale=np.std(x,axis=0,dtype=np.float64).clip(.01).tolist())


def design(x, scales):
    return ((np.asarray(x)-np.array(scales['mean']))/np.array(scales['scale'])).astype(np.float32)


def loss_rows(pred, y, mask):
    e=(pred-y).square();count=mask.sum(1)
    channel=(e*mask).sum(1)/count.clamp_min(1)
    valid=count>0
    activity=(channel[:,2:]*valid[:,2:]).sum(1)/valid[:,2:].sum(1).clamp_min(1)
    return dict(path=channel[:,0],body=channel[:,1],activity=activity,
                primary=(channel[:,0]+channel[:,1]+activity)/3)


def measure(pred, y, mask, stats):
    rows={k:v.numpy() for k,v in loss_rows(torch.tensor(pred,dtype=torch.float64),
        torch.tensor(y,dtype=torch.float64),torch.tensor(mask)).items()}
    true_delta=np.diff(y[...,0],axis=1)*stats['scale'][0]
    pred_delta=np.diff(pred[...,0],axis=1)*stats['scale'][0]
    rows['change1']=np.mean(((pred_delta-true_delta)/stats['delta_scale'])**2,axis=1)
    rows['close_mae_bps']=np.abs(pred[...,0]-y[...,0]).mean(1)*stats['scale'][0]*100
    a,b=true_delta.flatten(),pred_delta.flatten();sa,sb=float(a.std()),float(b.std())
    channels=[]
    for j in range(7):
        ok=mask[...,j];truth=y[...,j][ok];estimate=pred[...,j][ok]
        if not len(truth):
            channels.append(dict(support=0,mse=None,r2=None));continue
        variance=float(np.var(truth,dtype=np.float64));mse=float(np.mean((truth-estimate).astype(float)**2))
        channels.append(dict(support=int(ok.sum()),mse=mse,r2=1-mse/variance if variance>1e-12 else None))
    result=dict(metrics={k:float(v.mean()) for k,v in rows.items()},channels=channels,
        change1=dict(correlation=float(np.corrcoef(a,b)[0,1]) if min(sa,sb)>1e-12 else None,
                     std_ratio=sb/sa if sa>1e-12 else None))
    if not all(np.isfinite(v).all() for v in rows.values()):
        raise ValueError('Nonfinite readout metric')
    return result,rows


def ridge_candidates(x, y, mask, alphas=ALPHAS, device='cuda'):
    """Masked, centered ridge; missing labels are omitted, never fitted as zero.

    Solve all columns sharing a support pattern together. The unpenalized intercept
    is removed by support-specific centering; lambda scales mean squared error.
    """
    if min(alphas)<=0:raise ValueError('Positive ridge penalties required')
    flat=y.reshape(len(y),-1);valid=mask.reshape(len(y),-1)
    patterns,groups=np.unique(valid.T,axis=0,return_inverse=True)
    a=torch.tensor(x,dtype=torch.float64,device=device)
    result={alpha:dict(coef=np.zeros((x.shape[1],flat.shape[1])),bias=np.zeros(flat.shape[1])) for alpha in alphas}
    for group,ok in enumerate(patterns):
        columns=np.flatnonzero(groups==group)
        if ok.sum()<2:raise ValueError('Insufficient ridge support')
        use=torch.tensor(np.flatnonzero(ok),device=device)
        z=a[use];b=torch.tensor(flat[ok][:,columns],dtype=torch.float64,device=device)
        mx,my=z.mean(0),b.mean(0);z=z-mx;b=b-my
        gram=z.T@z/len(z);rhs=z.T@b/len(z)
        eigen,vectors=torch.linalg.eigh(gram);rot=vectors.T@rhs
        for alpha in alphas:
            coef=vectors@(rot/(eigen.clamp_min(0)[:,None]+alpha));bias=my-mx@coef
            result[alpha]['coef'][:,columns]=coef.cpu().numpy();result[alpha]['bias'][columns]=bias.cpu().numpy()
    return result


def ridge_predict(fit, x):
    return (np.asarray(x,dtype=np.float64)@fit['coef']+fit['bias']).reshape(-1,HISTORY,7).astype(np.float32)


class Probe(nn.Module):
    def __init__(self,width,seed,hidden=256):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,HISTORY*7))

    def forward(self,x):
        return self.net(x).reshape(-1,HISTORY,7)


def run_epoch(model, data, batch, device, opt=None, seed=0):
    n=len(data['x']);training=opt is not None;model.train(training)
    ids=np.random.default_rng(seed).permutation(n) if training else np.arange(n)
    totals={};steps=0
    with torch.set_grad_enabled(training):
        for left in range(0,n,batch):
            use=ids[left:left+batch];b={k:torch.tensor(np.asarray(v[use]),device=device) for k,v in data.items()}
            pred=model(b['x']);rows=loss_rows(pred,b['y'],b['mask'])
            if any(not torch.isfinite(v).all() for v in rows.values()):raise ValueError('Nonfinite probe objective')
            if training:
                opt.zero_grad(set_to_none=True);rows['primary'].mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();steps+=1
            for k,v in rows.items():totals[k]=totals.get(k,0.)+float(v.detach().sum())
    return {k:v/n for k,v in totals.items()},steps


@torch.no_grad()
def predictions(model,x,batch,device):
    model.eval()
    return np.concatenate([model(torch.tensor(np.asarray(x[i:i+batch]),device=device)).cpu().numpy() for i in range(0,len(x),batch)])
