"""History-only lumped reactive finite-volume expert. No state-delta head."""
import json, math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

R = 8.31446261815324
TREF = 298.15
DT = 0.00025
OUTPUTS = 22


class ReactiveStep(nn.Module):
    """Eight molar pools + three momenta + total internal/kinetic energy.

    The other pool is a learned calorically-perfect closure, not recovered
    chemistry. Shared reservoir exchange approximates unresolved transverse
    transport using ONLY the observed history. Signed exported species are
    retained; we do not claim species positivity or elemental conservation.
    """
    def __init__(self, normalization):
        super().__init__()
        source=json.loads((Path(__file__).parent/'THERMO_SOURCE.json').read_text())
        self.register_buffer('hf7',torch.tensor([s['hf_J_mol'] for s in source['species']]))
        self.register_buffer('m7',torch.tensor([16.043,28.010,44.009,18.015,16.023,17.031,17.007])*1e-3)
        self.register_buffer('mean',torch.tensor(normalization['mean']))
        self.register_buffer('std',torch.tensor(normalization['std']))
        # Unknown caloric properties are learned globally, not provided labels.
        init=torch.tensor([3.5,2.5,4.,3.5,3.,3.5,2.5,2.5])
        self.cv_raw=nn.Parameter(torch.log(torch.expm1(init-0.5)))
        self.other_m_raw=nn.Parameter(torch.tensor(math.log(28./30.)))
        self.other_h_raw=nn.Parameter(torch.tensor(0.))
        # Export sign is not certified by original 2022.1 export script.
        # One global calibration only; never a pixelwise free heat source.
        self.heat_sign_raw=nn.Parameter(torch.tensor(math.atanh(-0.5)))
        self.last_diagnostics={}

    def properties(self, reference):
        cv=R*(0.5+F.softplus(self.cv_raw))
        m=torch.cat([self.m7,0.030*torch.exp(self.other_m_raw.clamp(-2,2)).reshape(1)])
        hf=torch.cat([self.hf7,(1e5*torch.tanh(self.other_h_raw)).reshape(1)])
        # e(T)=hf(Tref)-R*Tref+cv*(T-Tref).
        e0=hf-(R+cv)*TREF
        return tuple(v.to(reference) for v in (m,cv,hf,e0))

    def observation_chart(self,x):
        # Repair only physically invalid temperatures inside the solver chart.
        # Encoder inputs and target/exported labels are never rewritten.
        # Each replacement uses valid neighbours of the same observed frame.
        b,h,w,nt,nc=x.shape
        t=x[...,10].permute(0,3,1,2).reshape(b*nt,1,h,w)
        valid=t>1.0
        counts=F.avg_pool2d(valid.to(t.dtype),3,1,1)
        sums=F.avg_pool2d(torch.where(valid,t,torch.zeros_like(t)),3,1,1)
        fallback=self.mean[10].clamp_min(1.0)
        replacement=torch.where(counts>0,sums/counts.clamp_min(1e-12),fallback)
        fixed=torch.where(valid,t,replacement).reshape(b,nt,h,w).permute(0,2,3,1)
        result=torch.cat([x[...,:10],fixed[...,None],x[...,11:]],-1)
        return result,(~valid).to(t.dtype).mean()

    def conserved(self, x):
        m,cv,hf,e0=self.properties(x)
        n=x[...,0:1]/(R*x[...,10:11])
        c7=n*x[...,2:9]
        c=torch.cat([c7,n-c7.sum(-1,keepdim=True)],-1)
        rho=(c*m).sum(-1,keepdim=True)
        velocity=x[...,11:14]
        momentum=rho*velocity
        energy=(c*(cv*x[...,10:11]+e0)).sum(-1,keepdim=True)+0.5*rho*velocity.square().sum(-1,keepdim=True)
        return torch.cat([c,momentum,energy],-1)

    def decode(self, q):
        m,cv,hf,e0=self.properties(q)
        c=q[...,:8];n=c.sum(-1,keepdim=True)
        rho=(c*m).sum(-1,keepdim=True)
        velocity=q[...,8:11]/rho.clamp_min(1e-12)
        capacity=(c*cv).sum(-1,keepdim=True)
        sensible=q[...,11:12]-0.5*rho*velocity.square().sum(-1,keepdim=True)-(c*e0).sum(-1,keepdim=True)
        temperature=sensible/capacity.clamp_min(1e-12)
        return n,rho,velocity,capacity,temperature

    def admissible(self,q):
        n,rho,u,cap,t=self.decode(q)
        return (torch.isfinite(q).all(-1,keepdim=True)&(n>1e-7)&(rho>1e-8)&(cap>1e-7)&(t>1.0))

    def limited(self,q,change,global_limit=False):
        # A shared scalar scales all conserved changes, preserving their
        # balances. Global limiter for spatial fluxes preserves cancellation.
        with torch.no_grad():
            def good(a):
                v=self.admissible(q+a*change)
                return v.all((1,2,3),keepdim=True) if global_limit else v
            shape=(len(q),1,1,1) if global_limit else (*q.shape[:-1],1)
            low=q.new_zeros(shape);high=q.new_ones(shape)
            whole=good(high)
            for _ in range(10):
                mid=(low+high)/2;ok=good(mid)
                low=torch.where(ok,mid,low);high=torch.where(ok,high,mid)
            alpha=torch.where(whole,torch.ones_like(low),0.99*low)
        return q+alpha*change,alpha

    def reaction_change(self,q,raw):
        m,cv,hf,e0=self.properties(q)
        n=q[...,:8].sum(-1,keepdim=True)
        d7=n*self.std[2:9]*0.5*torch.tanh(raw[...,8:15])
        # Background counter-source preserves mass, no independent source head.
        db=-(d7*m[:7]).sum(-1,keepdim=True)/m[7]
        dc=torch.cat([d7,db],-1)
        return torch.cat([dc,torch.zeros_like(q[...,8:])],-1)

    def reaction(self,q,raw):
        return self.limited(q,self.reaction_change(q,raw))

    def heat_release(self,q,raw):
        _,_,hf,_=self.properties(q)
        _,alpha=self.reaction(q,raw)
        omega=self.reaction_change(q,raw)[...,:8]*alpha/DT
        return (2*torch.tanh(self.heat_sign_raw))*(omega*hf).sum(-1,keepdim=True)

    def fields(self,q,raw,pressure_offset):
        n,rho,u,cap,t=self.decode(q)
        p=R*n*t
        heat=self.heat_release(q,raw)
        x=q[...,:7]/n
        # Diagnostic readout only, not an added correction to evolved state.
        speed=torch.linalg.vector_norm(u,dim=-1,keepdim=True)+self.std[14]*raw[...,21:22]
        return torch.cat([p,heat,x,p-pressure_offset,t,u,speed],-1)

    def exchange(self,q,raw,reservoirs,fraction):
        # Net in/out transverse exchange of the same complete conserved packet.
        weights=torch.softmax(raw[...,5:8],-1)
        reservoir=sum(reservoirs[k]*weights[...,k:k+1] for k in range(3))
        amount=1-torch.exp(-fraction*F.softplus(raw[...,4:5]-5.0))
        result=q+amount*(reservoir-q)
        return result,amount

    def spatial(self,q,raw,fraction):
        """Nonperiodic open extrapolation. Face flux shared by adjacent cells.

        Courant fluxes and integrated tractions are learned interval quantities;
        no asserted velocity-axis mapping or hidden dx/dt coefficients.
        """
        n,rho,u,cap,t=self.decode(q)
        adv=0.15*torch.tanh(raw[...,:2])
        mixing=0.02*torch.sigmoid(raw[...,2:3]-5)
        conductivity=0.02*torch.sigmoid(raw[...,3:4]-5)
        traction=0.2*rho[...,None]*self.std[11:14]*torch.tanh(raw[...,15:21].reshape(*raw.shape[:-1],2,3))
        # traction B,H,W,2,3; each row is an integrated momentum face flux.
        change=torch.zeros_like(q)
        for axis in [1,2]:
            direction=axis-1
            def pairs(v):
                left=torch.cat([v.select(axis,0).unsqueeze(axis),v],axis)
                right=torch.cat([v,v.select(axis,v.shape[axis]-1).unsqueeze(axis)],axis)
                return left,right
            ql,qr=pairs(q)
            vl,vr=pairs(adv[...,direction:direction+1]);v=(vl+vr)/2
            flux=v*torch.where(v>=0,ql,qr)
            dl,dr=pairs(mixing);flux=flux-(dl+dr)/2*(qr-ql)
            sl,sr=pairs(traction[...,direction,:]);stress=(sl+sr)/2
            ul,ur=pairs(u)
            tl,tr=pairs(t);cl,cr=pairs(cap);kl,kr=pairs(conductivity)
            heat=-(kl+kr)/2*(cl+cr)/2*(tr-tl)
            work=(stress*(ul+ur)/2).sum(-1,keepdim=True)
            flux=flux+torch.cat([torch.zeros_like(flux[...,:8]),stress,work+heat],-1)
            hi=flux.narrow(axis,1,q.shape[axis]);lo=flux.narrow(axis,0,q.shape[axis])
            change=change-(hi-lo)*fraction
        return self.limited(q,change,global_limit=True)

    def reservoirs(self,history_physical):
        packets=self.conserved(history_physical)
        return [packets[...,:3,:].mean(-2),packets[...,3:6,:].mean(-2),packets[...,6:,:].mean(-2)]

    def forward(self,history,query):
        x=history*self.std+self.mean
        x,repair_fraction=self.observation_chart(x)
        q=self.conserved(x[...,-1,:])
        reservoirs=self.reservoirs(x)
        offset=(x[...,-1,0:1]-x[...,-1,9:10]).mean((1,2),keepdim=True)
        raw0=query(history[...,-1,:])
        q,sp0=self.spatial(q,raw0,0.5)
        q,ex0=self.exchange(q,raw0,reservoirs,0.5)
        stage=(self.fields(q,raw0,offset)-self.mean)/self.std
        raw1=query(stage)
        q,reaction_alpha=self.reaction(q,raw1)
        q,ex1=self.exchange(q,raw1,reservoirs,0.5)
        q,sp1=self.spatial(q,raw1,0.5)
        endpoint=(self.fields(q,raw1,offset)-self.mean)/self.std
        raw2=query(endpoint)
        result=(self.fields(q,raw2,offset)-self.mean)/self.std
        self.last_diagnostics={
            'invalid_observed_temperature_fraction':float(repair_fraction.detach()),
            'reaction_limited_fraction':float((reaction_alpha<0.999).float().mean().detach()),
            'spatial_min_alpha':float(torch.minimum(sp0.min(),sp1.min()).detach()),
            'exchange_mean':float((ex0+ex1).mean().detach()),
            'heat_export_scale':float((2*torch.tanh(self.heat_sign_raw)).detach()),
        }
        return result
