"""Validation-only convergence control; serializable at every checkpoint."""
import math

class Convergence:
    keys=('rollout_l2re','rollout_nmse','fixed_train_nmse')
    def __init__(self,cfg,state=None):
        self.cfg=cfg
        self.state=state or {'lr':cfg['lr_peak'],'anchors':{},'stale':0,'reductions':0,'converged':False,'last_step':None,'events':[]}
    def observe(self,step,metrics):
        assert self.state['last_step'] is None or step>self.state['last_step'],'duplicate or reversed validation'
        assert all(math.isfinite(metrics[k]) and metrics[k]>=0 for k in self.keys),'invalid convergence metric'
        initial=not self.state['anchors'];improved=[]
        for k in self.keys:
            old=self.state['anchors'].get(k)
            if old is None or metrics[k]<old*(1-self.cfg['min_relative_improvement']):
                self.state['anchors'][k]=metrics[k];improved.append(k)
        self.state['stale']=0 if improved else self.state['stale']+1
        event={'step':step,'metrics':dict(metrics),'significantly_improved':improved,'action':'initial' if initial else 'continue'}
        if step>=self.cfg['min_total_updates'] and self.state['stale']>=self.cfg['plateau_patience']:
            if self.state['lr']>self.cfg['lr_min']*(1+1e-9):
                self.state['lr']=max(self.cfg['lr_min'],self.state['lr']*self.cfg['lr_factor'])
                self.state['reductions']+=1;self.state['stale']=0;event['action']='reduce_lr'
            else:
                self.state['converged']=True;event['action']='converged'
        self.state['last_step']=step;event.update(lr=self.state['lr'],stale=self.state['stale']);self.state['events'].append(event)
        return event

    def learning_rate(self,step):
        warm=self.cfg['warmup_updates'];t=step-self.cfg['source_step']+1
        if t<=warm:return min(self.state['lr'],1e-6+(self.cfg['lr_peak']-1e-6)*t/warm)
        return self.state['lr']

    def validation_due(self,step):
        last=self.state['last_step']
        return last is None or (step%self.cfg['validation_every']==0 and step>last)
