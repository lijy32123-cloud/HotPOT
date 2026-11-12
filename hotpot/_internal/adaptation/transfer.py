from __future__ import annotations


from contextlib import contextmanager


import hashlib


import json


import math


from hotpot.protocols import COEFFICIENT_SHOTS, WAVE_SHOTS, maximum_learning_rate, training_seed

VERSION = "hotpot.release.v2"


KNOWN = (
    "ns2d_fno_1e-5", "ns2d_fno_1e-3", "ns2d_pdb_M1e-1_eta1e-2_zeta1e-2",
    "swe_pdb", "dr_pdb", "cfdbench",
)


FEWSHOT = ("fno_1e-4", "cns_m1_eta1e-2")


CFDB = ("cfdb_cavity", "cfdb_cylinder", "cfdb_tube")


WAVE = ("Wave-Layer", "Wave-Gauss")


SHOTS = tuple(sorted(set(COEFFICIENT_SHOTS + WAVE_SHOTS)))


MODES = ("known-train", "known-test", "fewshot-train", "fewshot-test",
         "zero-shot", "wave-train", "wave-test")


SNAPSHOT_SCHEMA = "hotpot.final_validation.transfer.full_state.v1"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def recipe(a, source_sha):
    return {"version": VERSION, "task": a.task, "source_key": a.source_key,
            "source_sha256": source_sha, "arm": a.arm, "shots": a.shots,
            "target_replicate": 0, "training_rng_seed": training_seed(a.mode),
            "source_seed": 20260815, "selection": a.selection,
            "max_learning_rate": maximum_learning_rate(a.mode, a.task, a.shots),
            "validation_metric": ("full_rollout_per_channel_L2RE" if a.mode.startswith("fewshot-") else
                                  "mean_per_step_L2RE" if a.mode.startswith("wave-") else None),
            "fno_execution_substeps": 8,
            "epochs": 200 if a.mode.startswith("known-") else 500,
            "scope": "SMOKE" if a.smoke_updates else "FINAL",
            "smoke_updates": a.smoke_updates}


def validate_wave_split(manifest, task):
    """Original custom split, seed0 only; validate identities, not array contents."""
    if manifest.get("split_id") != "HETPOT_WAVE_CUSTOM_SPLIT_20260915_V1":
        raise ValueError("Wave split version is not the original successful custom split")
    d = manifest["datasets"][task]
    pool = d["train_pool"]["indices"]
    test = d["test"]["indices"]
    if (len(pool) != 1800 or len(set(pool)) != 1800 or len(test) != 200 or
            len(set(test)) != 200 or set(pool) & set(test) or
            set(pool) | set(test) != set(range(2000)) or d["test"]["source"] != "train" or
            d["validation"] != {"source": "valid", "indices": list(range(100))}):
        raise ValueError("Wave train/validation/test split drift")
    excluded = set(d["excluded_from_test"])
    prefix = f'{manifest["split_id"]}|{task}|holdout|{manifest["split_seed"]}|'
    order = sorted(set(range(2000)) - excluded,
                   key=lambda i: (hashlib.sha256(f"{prefix}{i}".encode()).hexdigest(), i))
    if set(test) != set(order[:200]):
        raise ValueError("Wave heldout rows differ from the original split rule")
    prefix = f'{manifest["split_id"]}|{task}|fewshot|0|'
    order = sorted(pool, key=lambda i: (hashlib.sha256(f"{prefix}{i}".encode()).hexdigest(), i))
    for n in WAVE_SHOTS:
        if d["fewshot_train_indices"]["0"][str(n)] != order[:n]:
            raise ValueError("Wave seed0 nested training subsets drift")
    # Source paths saved in the historical manifest are never followed.
    return d


@contextmanager
def patched(module, **replacements):
    """Restore process-local injection even after a legacy main raises."""
    missing = object()
    old = {key: getattr(module, key, missing) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(module, key, value)
        yield module
    finally:
        for key, value in old.items():
            if value is missing:
                delattr(module, key)
            else:
                setattr(module, key, value)


def finite_tree(value, path="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite metric: " + path)
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, path + "." + str(index))


def verify_snapshot(snapshot, expected_recipe):
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("recipe") != expected_recipe:
        raise ValueError("target checkpoint version/source/task/arm/subset/selection/scope mismatch")
    if snapshot.get("test_used_for_selection") is not False:
        raise ValueError("target checkpoint was not selected independently of test")
    if not isinstance(snapshot.get("full_model_state"), dict) or not snapshot["full_model_state"]:
        raise ValueError("target checkpoint requires full model state, including buffers")
    epoch = snapshot.get("completed_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= expected_recipe["epochs"]:
        raise ValueError("invalid selected epoch")
    if expected_recipe["selection"] == "fixed-final" and epoch != expected_recipe["epochs"]:
        raise ValueError("fixed-final snapshot is not the requested endpoint")
    if expected_recipe["epochs"] == 500:
        per_epoch = math.ceil(expected_recipe["shots"] / min(20, expected_recipe["shots"]))
        if snapshot.get("updates") != epoch * per_epoch:
            raise ValueError("selected target update count differs from the complete epoch")
        if expected_recipe["selection"] == "original-val" and epoch % 10:
            raise ValueError("original validation selection requires an evaluated multiple of ten epochs")
