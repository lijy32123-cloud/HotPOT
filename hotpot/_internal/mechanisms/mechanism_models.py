from pathlib import Path
import sys,types
import torch
from torch import nn

from hotpot._internal.data_protocol.manifest import CONDITION_FIELDS
from hotpot._internal.data_protocol.native_charts import fno_native64_exact_left_inverse
from hotpot._internal.encoder.model import EVIDENCE_SELECTED_INTERFACE_REVISION
from hotpot._internal.encoder.semantic_chart import SemanticHistory,SEMANTIC_FEATURES
from hotpot._internal.foundation.model import SemanticFieldProgramFoundationS90M
from hotpot._internal.foundation.base6 import Base6SemanticFieldFoundationS90M
from hotpot._internal.foundation.config import tiny_17m_config
from hotpot._internal.controls.models import V13DirectStateControl

ARMS=('full','feedback','no_history_derivatives','direct')
REMOVED_FEATURES=SEMANTIC_FEATURES[1:6]

def residual_refine(current,query):
    for _ in range(4):
        first=query(current)
        second=query(current+first/8.)
        current=current+second/4.
    return current

def feedback_forward(self,history,decoded,conditions,condition_mask):
    column=CONDITION_FIELDS.index('observation_dt')
    if not bool(((conditions[:,column]==1.) & (condition_mask[:,column]==1.)).all()):
        raise ValueError('NS feedback control is defined for the source unit saved interval')
    self.head.reset_calls()
    native=fno_native64_exact_left_inverse(history[...,0].permute(0,3,1,2).contiguous())
    out=residual_refine(native[:,-1:].contiguous(),lambda q:self.head(decoded,q,conditions,condition_mask))
    if self.head.call_count!=8:raise RuntimeError('NS feedback query count drift')
    return out

class FeedbackModel(SemanticFieldProgramFoundationS90M):
    """Only NS execution changes; DR and every other family remain exact Full."""
    def __init__(self,config=None):
        super().__init__(config or tiny_17m_config(),interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION)
        self.programs['fno'].forward=types.MethodType(feedback_forward,self.programs['fno'])

class NoHistoryDerivatives(SemanticFieldProgramFoundationS90M):
    """Remove five history-chart differences, not coefficients or NS stage charts."""
    def __init__(self,config=None):
        super().__init__(config or tiny_17m_config(),interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION)
    def _semantic_batch(self,history,grouped):
        value,indices=super()._semantic_batch(history,grouped)
        keep=value.features.new_ones(1,1,1,len(SEMANTIC_FEATURES),1,1)
        keep[:,:,:,1:6]=0
        return SemanticHistory(value.features*keep,value.field_mask,value.field_mean,value.field_log_scale,value.relative_lags),indices

def build(arm,config=None):
    resolved=config or tiny_17m_config()
    if arm=='full':return SemanticFieldProgramFoundationS90M(resolved,interface_revision=EVIDENCE_SELECTED_INTERFACE_REVISION)
    if arm=='feedback':return FeedbackModel(resolved)
    if arm=='no_history_derivatives':return NoHistoryDerivatives(resolved)
    if arm=='direct':return V13DirectStateControl(resolved)
    raise ValueError(arm)

def dr_generic(current,query):
    first_a,first_b=query(current);first=first_a+first_b
    second_a,second_b=query(current+.5*first)
    return current+.5*(first+second_a+second_b)

class Base6FinalRound(Base6SemanticFieldFoundationS90M):
    arm='full'
    def __init__(self,workspace=None,config=None):
        nn.Module.__init__(self);self.workspace=workspace;self.model=build(self.arm,config)
