"""Public model builders for HotPOT and the paper's ablations."""
from pathlib import Path
import torch
from torch import nn

from ._internal.foundation.base6 import Base6SemanticFieldFoundationS90M
from ._internal.foundation.config import FoundationScaleConfig, tiny_17m_config
from ._internal.mechanisms.mechanism_models import NoHistoryDerivatives, FeedbackModel
from ._internal.controls.models import Base6V13NoSharingControl, Base6V13DirectStateControl
from ._internal.late_sharing.models import Base6V13LateSharedEvolutionControl
from ._internal.history_features.nodiff_chart import install_no_difference_chart

VARIANTS = ('full', 'untied', 'late', 'late_untied', 'direct', 'generic_iteration')


class HotPOT(Base6SemanticFieldFoundationS90M):
    """Shared history encoder with PDE-specific neural-numerical experts.

    Input: [batch, 128, 128, 10, 4], padded to four field channels.
    Output: [batch, 128, 128, 1, 4]. Dataset routing is explicit.
    """
    def __init__(self, workspace=None, config=None, *, generic_iteration=False):
        nn.Module.__init__(self)
        self.workspace = workspace
        self.model = (FeedbackModel if generic_iteration else NoHistoryDerivatives)(
            config or tiny_17m_config())
        install_no_difference_chart(self)


def build_model(scale=17, variant='full', *, workspace=None, config=None, target=False):
    """Build a fresh model. ``target=True`` selects target-stage execution settings.

    ``scale`` is the paper's activated-parameter scale, not total storage.
    Explicit ``config`` is useful for CPU smoke tests.
    """
    if scale not in (17, 90):
        raise ValueError('scale must be 17 or 90')
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    if variant != 'full' and scale != 17:
        raise ValueError('The paper ablations use the 17M scale')
    config = config or (tiny_17m_config() if scale == 17 else FoundationScaleConfig())
    if variant in ('full', 'generic_iteration'):
        model = HotPOT(workspace, config, generic_iteration=variant == 'generic_iteration')
    elif variant == 'late_untied':
        from ._internal.late_untied.models import model_class
        model = model_class(None, None, config=config)(workspace=workspace)
    else:
        constructor = {'untied': Base6V13NoSharingControl,
                       'direct': Base6V13DirectStateControl,
                       'late': Base6V13LateSharedEvolutionControl}[variant]
        model = constructor(workspace=workspace, config=config)
    install_no_difference_chart(model)
    if target:
        if variant != 'full':
            raise ValueError('Target transfer is defined for the full model')
        model.model.programs['fno'].substeps = 8
    return model


EXPERT_GROUPS = ('private_mixers', 'action_context_decoders', 'programs')
EXPERT_FAMILIES = ('fno', 'cns', 'swe', 'dr', 'cfdb')


def load_checkpoint(model, path):
    """Strictly load a tensor state or a training/transfer checkpoint.

    Loading uses ``weights_only=True``. Optimizer state is not needed for inference.

    A single-expert file (``meta['single_expert']`` names a PDE family) stores the shared
    history encoder and that family's expert only. Its tensors must match the model exactly;
    the experts of the other families are absent from the file, keep their initial values and
    are never executed for inputs of the stored family.
    """
    snapshot = torch.load(Path(path), map_location='cpu', weights_only=True)
    for key in ('full_model_state', 'model', 'model_state', 'state_dict'):
        if isinstance(snapshot, dict) and key in snapshot and isinstance(snapshot[key], dict):
            state = snapshot[key]
            break
    else:
        state = snapshot
    meta = snapshot.get('meta') if isinstance(snapshot, dict) else None
    family = meta.get('single_expert') if isinstance(meta, dict) else None
    if family is None:
        model.load_state_dict(state, strict=True)
        return snapshot
    if family not in EXPERT_FAMILIES:
        raise ValueError(f'unknown expert family: {family}')
    result = model.load_state_dict(state, strict=False)

    def other_expert(name):
        return any(name.startswith(f'model.{group}.') and not name.startswith(f'model.{group}.{family}.')
                   for group in EXPERT_GROUPS)

    if result.unexpected_keys or not all(other_expert(name) for name in result.missing_keys):
        raise RuntimeError('single-expert file does not match the shared encoder and the selected expert')
    return snapshot
