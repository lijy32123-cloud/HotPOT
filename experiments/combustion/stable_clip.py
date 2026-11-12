import torch

@torch.no_grad()
def clip_grad_norm(parameters,max_norm=1.):
    parameters=list(parameters)
    grads=[p.grad for p in parameters if p.grad is not None]
    total=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(),2) for g in grads]),2)
    if torch.isfinite(total):return torch.nn.utils.clip_grad_norm_(parameters,max_norm)
    if not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('Nonfinite gradient tensor')
    total64=torch.stack([g.detach().double().square().sum() for g in grads]).sum().sqrt()
    if not bool(torch.isfinite(total64)):raise FloatingPointError('Nonfinite float64 gradient norm')
    factor=(max_norm/(total64+1e-6)).clamp(max=1.)
    for g in grads:g.mul_(factor.to(g.dtype))
    return total64
