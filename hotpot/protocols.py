"""Published experiment settings, shared by execution and checkpoint metadata."""

COEFFICIENT_SHOTS = (16, 64, 256)
WAVE_SHOTS = (16, 32, 64, 128)
KNOWN_MAX_LR = {
    'ns2d_fno_1e-5': 1e-5,
    'ns2d_fno_1e-3': 1e-3,
    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2': 1e-4,
    'swe_pdb': 1e-3,
    'dr_pdb': 1e-3,
    'cfdbench': 1e-5,
}


def training_seed(mode):
    # Replicate index 0 in adaptation corresponds to this RNG seed.
    return 20260815 if mode.startswith('known-') else 20260831


# Few-shot maximum learning rates, selected for each target and budget from
# FEWSHOT_LR_CANDIDATES by the lowest full-rollout validation error.
FEWSHOT_LR_CANDIDATES = (1e-3, 1e-4, 1e-5)
FEWSHOT_MAX_LR = {
    'fno_1e-4': {16: 1e-4, 64: 1e-4, 256: 1e-5},
    'cns_m1_eta1e-2': {16: 1e-3, 64: 1e-3, 256: 1e-3},
}


def maximum_learning_rate(mode, task, shots=None):
    if mode.startswith('known-'):
        return KNOWN_MAX_LR[task]
    if mode.startswith('fewshot-'):
        return FEWSHOT_MAX_LR.get(task, {}).get(shots, 1e-4)
    return 1e-4


def normalize_arguments(a):
    if a.mode.startswith('known-'):
        if a.task not in KNOWN_MAX_LR or a.arm != 'transfer' or a.shots is not None:
            raise ValueError('known modes require a pre-training task, transfer arm, and no shots')
        a.selection = 'fixed-final'
    elif a.mode.startswith('fewshot-'):
        if a.task not in ('fno_1e-4', 'cns_m1_eta1e-2') or a.shots not in COEFFICIENT_SHOTS:
            raise ValueError('NS/CNS few-shot modes require shots 16, 64, or 256')
        if a.selection != 'original-val':
            raise ValueError('NS/CNS few-shot uses minimum full-rollout validation')
    elif a.mode.startswith('wave-'):
        if a.task not in ('Wave-Layer', 'Wave-Gauss') or a.shots not in WAVE_SHOTS or a.scale != 17:
            raise ValueError('Wave requires scale 17 and shots 16, 32, 64, or 128')
        if a.selection != 'original-val':
            raise ValueError('Wave uses minimum rollout validation')
    elif a.task not in ('fno_1e-4', 'cns_m1_eta1e-2', 'cfdb_cavity', 'cfdb_cylinder', 'cfdb_tube') or a.arm != 'transfer' or a.shots is not None:
        raise ValueError('zero-shot requires a coefficient/CFDB task, transfer arm, and no shots')
    if a.mode.endswith('-test') and a.target_checkpoint is None:
        raise ValueError('Test modes require --target-checkpoint')
    return a
