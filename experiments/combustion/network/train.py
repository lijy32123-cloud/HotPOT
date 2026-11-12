import argparse,json,os,random,signal,time,traceback
import numpy as np
import torch
from common import BASE,IntensityData,Sampler,predict,evaluate,digest,atomic_json,save
from intensity_models import build,state_hash

STOP=False
def stop(*_):
    global STOP
    STOP=True

def rng_state():
    return {'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
def restore_rng(s):
    torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['hotpot_relaxed']);parser.add_argument('--lr',type=float,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    cfg=json.loads((BASE/'PROTOCOL.json').read_text());code=json.loads((BASE/'CODE_LOCK.json').read_text())
    for name,sha in code.items():assert digest(BASE/name)==sha,('changed executable',name)
    assert json.loads((BASE/'CPU_CHECK.json').read_text())['status']=='PASS'
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model,ledger=build(args.arm,cfg['seed']);model.cuda()
    assert ledger==json.loads((BASE/'CPU_CHECK.json').read_text())['models'][args.arm]['ledger']
    data=IntensityData('cuda');sampler=Sampler(len(data.rows['train']),cfg['seed'])
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=tuple(cfg['betas']),weight_decay=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_updates'])
    out=BASE/('smoke' if args.smoke else 'runs')/args.arm/f'lr_{args.lr:g}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    atomic_json(out/'PROVENANCE.json',{'protocol':cfg,'ledger':ledger,'code':code,'host':os.uname().nodename,'gpu':torch.cuda.get_device_name(),'pid':os.getpid(),'learning_rate':args.lr})
    step=0;best={'rmse':float('inf'),'step':None};elapsed=0.
    last=out/'last.pt'
    def load_state(s):
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);sampler.load_state_dict(s['sampler']);restore_rng(s)
    if last.exists() and not args.smoke:
        s=torch.load(last,map_location='cpu',weights_only=False)
        assert s['code']==code and s['ledger']==ledger and s['lr']==args.lr
        load_state(s);step=s['step'];best=s['best'];elapsed=s['elapsed_seconds'];del s
    segment_start=time.time();last_mark=segment_start
    def snapshot():
        return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'sampler':sampler.state_dict(),**rng_state(),'code':code,'ledger':ledger,'lr':args.lr,'step':step,'best':best,'elapsed_seconds':elapsed+time.time()-segment_start}
    def update():
        model.train();opt.zero_grad(set_to_none=True)
        chosen=sampler.next(cfg['global_batch']);history,truth=data.batch('train',chosen.to(data.device))
        pred=predict(model,history);mse=(pred-data.target(truth)).square().mean();loss=mse/16
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite I loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
        opt.step();scheduler.step()
        return float(mse.detach()),float(norm.detach()),chosen.tolist()
    total=5 if args.smoke else cfg['max_updates'];timings=[]
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'event':'start','arm':args.arm,'lr':args.lr,'smoke':args.smoke,'step':step,'parameters':ledger['stored_parameters']}),flush=True)
    for step in range(step+1,total+1):
        begin=time.time();mse,norm,chosen=update();torch.cuda.synchronize();timings.append(time.time()-begin)
        row={'step':step,'I_normalized_mse':mse,'official_16_slot_mse':mse/16,'gradient_norm_before_clip':norm,'lr_next':scheduler.get_last_lr()[0],'numerical':model.diagnostics,'seconds':timings[-1],'time':time.time(),'epoch':sampler.epoch,'batch_indices':chosen,'elapsed_seconds':elapsed+time.time()-segment_start}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step<=5 or step%20==0:
            atomic_json(out/'PROGRESS.json',{'phase':'training',**row,'median_update_seconds':float(np.median(timings[-50:]))});print(json.dumps(row),flush=True)
        if step==1 or step%cfg['checkpoint_every']==0 or STOP or step==total:save(last,snapshot())
        if STOP:
            atomic_json(out/'PAUSED.json',{'step':step,'time':time.time(),'checkpoint':str(last)});return 75
        if (args.smoke and step==total) or (not args.smoke and step%cfg['validation_every']==0):
            val_start=time.time()
            def progress(done,n):
                if done%512==0 or done==n:atomic_json(out/'PROGRESS.json',{'phase':'validation','step':step,'completed_windows':done,'total_windows':n,'time':time.time()})
            result=evaluate(model,data,limit=64 if args.smoke else None,batch=cfg['evaluation_batch'],progress=progress)
            result['step']=step;result['seconds']=time.time()-val_start
            atomic_json(out/f'validation_{step:05d}.json',result)
            summary={k:v for k,v in result.items() if k!='records'}
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
            if result['rmse']<best['rmse']:
                best={'rmse':result['rmse'],'relative_l2':result['relative_l2'],'step':step}
                save(out/'best.pt',{'model':model.state_dict(),'ledger':ledger,'code':code,'step':step,'lr':args.lr,'validation':summary})
            save(last,snapshot());print(json.dumps({'event':'validation',**summary}),flush=True)
        if args.smoke and step==total:
            # Optimizer step tensors can alias the loaded state; read the immutable file
            # afresh for each branch rather than reusing a possibly mutated dict.
            load_state(torch.load(last,map_location='cpu',weights_only=False))
            a=update();hash_a=state_hash(model)
            load_state(torch.load(last,map_location='cpu',weights_only=False))
            b=update();hash_b=state_hash(model)
            atomic_json(out/'RESUME_CHECK.json',{'first':a,'restored':b,'first_hash':hash_a,'restored_hash':hash_b,'exact':a==b and hash_a==hash_b})
            assert a==b and hash_a==hash_b,'resume must reproduce the next optimizer update exactly'
            atomic_json(out/'COMPLETE.json',{'status':'PASS','steps':step,'median_update_seconds':float(np.median(timings[1:])),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'resume_exact':True,'ledger':ledger,'validation_subset':summary,'time':time.time()})
            return 0
    atomic_json(out/'COMPLETE.json',{'status':'COMPLETE','steps':step,'best':best,'elapsed_seconds':elapsed+time.time()-segment_start,'gpu_hours':(elapsed+time.time()-segment_start)/3600,'median_update_seconds':float(np.median(timings[-100:])),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'time':time.time()})
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise
