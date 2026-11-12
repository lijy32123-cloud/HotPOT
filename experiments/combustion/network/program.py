"""Finite-interval nonnegative contribution transport and birth assembly.

The guarantees are discrete positivity, conservative transport and a complete
ledger. This is not a closed OH* chemistry law or identified gas velocity.
"""
import torch


def remap(mass, displacement):
    """Translate each uniform unit-cell contribution once, with open boundaries.

    Nonnegative overlap weights partition unity on the extended image grid.
    Crossing outside the field of view is recorded rather than wrapped.
    """
    b,t,h,w=mass.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=mass.device,dtype=mass.dtype),
        torch.arange(w,device=mass.device,dtype=mass.dtype),indexing='ij')
    x=xx+displacement[:,:,0];y=yy+displacement[:,:,1]
    ix=x.floor().long();iy=y.floor().long()
    fx=x-ix;fy=y-iy
    out=torch.zeros((b*t,h*w),device=mass.device,dtype=mass.dtype)
    escaped=mass.new_zeros(b,t)
    for dx,dy,weight in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        jx=ix+dx;jy=iy+dy
        inside=(jx>=0)&(jx<w)&(jy>=0)&(jy<h)
        contribution=mass*weight
        index=(jy.clamp(0,h-1)*w+jx.clamp(0,w-1)).reshape(b*t,-1)
        out=out.scatter_add(1,index,(contribution*inside).reshape(b*t,-1))
        escaped=escaped+(contribution*(~inside)).sum((-2,-1))
    return out.reshape(b,t,h,w),escaped


def assemble(initial, retention, generation, displacement):
    """I_k=T(D_k)(r_k I_0)+G_k, for each lead k in a 20-frame block.

    r is in [0,1], G>=0. Both describe complete-interval contributions, not
    frozen instantaneous reaction coefficients. No derivative or step size is
    inferred from unavailable exposure metadata.
    """
    retained=initial[:,None]*retention
    transported,escaped=remap(retained,displacement)
    result=transported+generation
    lost=(initial[:,None]*(1-retention)).sum((-2,-1))
    born=generation.sum((-2,-1))
    original=initial.sum((-2,-1))[:,None].expand_as(born)
    balance=result.sum((-2,-1))-(original-lost-escaped+born)
    return result,{'transported':transported,'escaped':escaped,'lost':lost,
        'born':born,'initial_total':original,'balance':balance}
