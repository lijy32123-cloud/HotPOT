from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys

from .core import atomic_json, checked_path, sha256
from .transfer import (CFDB, FEWSHOT, SNAPSHOT_SCHEMA, WAVE, digest_json, finite_tree,
                       patched, validate_wave_split, verify_snapshot)

from hotpot.protocols import maximum_learning_rate
from .coefficient_rollout import RolloutValidationDataset, rollout as coefficient_rollout

BASE = "hotpot._internal.data_protocol"
OLD = "hotpot._internal.encoder"
OOD = "hotpot._internal.coefficients"
WAVE_MODELS = "hotpot/_internal/wave/wave_models.py"


def module(name):
    return importlib.import_module(name)


def fno8(wrapper):
    """Only this target instance changes; source classes/files remain frozen."""
    program = wrapper.model.programs["fno"]
    if not hasattr(program, "substeps"):
        raise ValueError("selected Full model lacks the source-aligned FNO substep adapter")
    program.substeps = 8
    if program.substeps != 8:
        raise RuntimeError("target FNO execution must use 8 substeps")
    return wrapper


def full_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_tensor(path, payload):
    import torch
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checked_state(model, state):
    import torch
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("full state keys differ (including buffers)")
    for name, value in state.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or
                value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("invalid full-state tensor: " + name)
    model.load_state_dict(state, strict=True)


def identity(path, repo):
    if path.is_file():
        return {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size,
                "sha256": sha256(path)}
    # Directory evidence deliberately identifies file inventory, not content hashes.
    records = [(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns)
               for p in sorted(path.rglob("*.hdf5"))]
    return {"path": str(path.relative_to(repo)), "inventory_entries": len(records),
            "inventory_size_mtime_sha256": digest_json(records), "content_hashed": False}


def training_identity(a, prepared):
    return {key: identity(path, a.repo) for key, path in prepared["paths"].items()
            if not key.endswith((".test", "test_sidecar"))}


def source_inventory(repo):
    sources = {}
    for obj in tuple(sys.modules.values()):
        file = getattr(obj, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(repo) and path.is_file() and path.suffix == ".py":
                sources[str(path.relative_to(repo))] = sha256(path)
    return sources


def checkpoint_payload(model, prepared, epoch, updates, validation, data_identity):
    sources = source_inventory(prepared["repo"]) if "repo" in prepared else {}
    if "repo" in prepared and prepared["recipe"]["task"] in WAVE:
        sources[WAVE_MODELS] = sha256(prepared["repo"] / WAVE_MODELS)
    return {"schema": SNAPSHOT_SCHEMA, "recipe": prepared["recipe"],
            "completed_epoch": epoch, "updates": updates,
            "validation_l2re": validation, "data_identity": data_identity,
            "full_model_state": full_state(model), "test_used_for_selection": False,
            "source_sha256": sources}




def restoration(expected, observed, output):
    if not isinstance(expected, (int, float)) or not math.isfinite(expected):
        raise ValueError("saved validation metric is not finite")
    tolerance = 1e-7 + 1e-4 * abs(expected)
    good = math.isfinite(observed) and abs(observed - expected) <= tolerance
    atomic_json(output / "RESTORATION_GATE.json", {
        "status": "PASS" if good else "FAIL", "saved": expected,
        "observed": observed if math.isfinite(observed) else str(observed),
        "tolerance": tolerance, "test_accessed": False})
    if not good:
        raise ValueError("full-state validation restoration failed before test access")




def known_train(a, bindings, prepared, output, bound):
    """Call the successful corrected E200 main with one rank and exact B40/M20."""
    import torch
    from argparse import Namespace
    train = module(OLD + ".finetune_ddp")
    data = module(BASE + ".data")
    account = module("hotpot._internal.foundation.accounting")
    entries = tuple(replace(row, train_path=str(prepared["paths"][f"transfer.{a.task}.train"]))
                    if row.key == a.task else row for row in data.BASE6)
    if a.num_workers != 0:
        raise ValueError("known-train process-local data adapter requires --num-workers 0")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("one task per invocation: known-train uses one rank, global B40/micro B20")
    native_output = output / "native"
    data_id = training_identity(a, prepared)
    args = Namespace(workspace=a.repo, cfdb_sidecar=prepared["paths"]["transfer.cfdb.train_sidecar"],
                     base_checkpoint=prepared["source_path"], base_sha256=prepared["source_entry"]["sha256"],
                     output=native_output, target=a.task, epochs=200, warmup_epochs=40,
                     local_batch_size=40, num_workers=0,
                     learning_rate=maximum_learning_rate(a.mode, a.task),
                     checkpoint_every=10, max_updates=0, gradient_microbatch_size=20,
                     resume=None, smoke_updates=a.smoke_updates)
    def init():
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        torch.distributed.init_process_group("nccl", rank=0, world_size=1,
                                             init_method=(output / "rendezvous").as_uri())
        return 0, 0, 1, device
    def base_identity(model, path, *, expected_sha256):
        
        if model is not bound or path != prepared["source_path"] or sha256(path) != expected_sha256:
            raise ValueError("source changed before legacy fine-tuner injection")
        return {"path": str(path), "sha256": expected_sha256, "bytes": path.stat().st_size,
                "completed_epoch": 1000, "updates": 545000, "binding_key": a.source_key}
    replacements = dict(arguments=lambda: args, Base6SemanticFieldFoundation=lambda **kw: bound,
                        _load_base_checkpoint=base_identity, SEED=20260815, _init_distributed=init,
                        parameter_ledger=account.parameter_ledger,
                        _microbatch_slices=lambda size, micro: tuple(slice(i, min(i + micro, size))
                                                                    for i in range(0, size, micro)))
    try:
        with patched(data, BASE6=entries), patched(train, **replacements):
            train.main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    result = json.loads((native_output / "COMPLETE.json").read_text())
    expected_status = "SMOKE_COMPLETE" if a.smoke_updates else "E0200_COMPLETE"
    if result.get("status") != expected_status or result.get("target") != a.task:
        raise RuntimeError("legacy main did not complete the requested known-task scope")
    if a.smoke_updates and result["updates"] != a.smoke_updates:
        raise RuntimeError("legacy smoke update bound drift")
    if not a.smoke_updates and (result["completed_epoch"] != 200 or result["updates"] != result["target_updates"]):
        raise RuntimeError("known E200 update endpoint drift")
    snapshot = checkpoint_payload(bound, prepared, result["completed_epoch"], result["updates"], None, data_id)
    path = output / "target_full_state.pt"
    save_tensor(path, snapshot)
    return {"checkpoint": path.name, "checkpoint_sha256": sha256(path), "native_completion": result,
            "updates": result["updates"], "test_accessed": False}


def ood_data(a, prepared):
    data = module(OOD + ".data")
    spec = replace(data.TARGETS[a.task], relative_train_path=str(prepared["paths"][f"transfer.{a.task}.train"]))
    rows, valid = data.split_rows(spec, a.shots, 0)
    sampler = data.EpochBatchSampler(spec, rows, seed=0, batch_size=min(20, a.shots))
    dataset = data.TargetStorageDataset(a.repo, spec)
    from torch.utils.data import DataLoader
    train = DataLoader(dataset, batch_sampler=sampler, num_workers=a.num_workers)
    validation = DataLoader(RolloutValidationDataset(a.repo, spec, valid),
                            batch_size=8, num_workers=a.num_workers)
    return data, spec, sampler, train, validation, {
        "training_rows": list(rows), "validation_rows": list(valid),
        "validation_start": 0, "validation_horizon": 20 if spec.family == "fno" else 11}


def configure_ood(a, bound, spec):
    model = new_ours(a, bound).model
    prefixes = ("shared_analyzer.", f"private_mixers.{spec.family}.",
                f"action_context_decoders.{spec.family}.", f"programs.{spec.family}.")
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        if parameter.requires_grad:
            names.append(name)
    if not any(n.startswith(f"programs.{spec.family}.") for n in names):
        raise ValueError("empty target-private trainability")
    return model.cuda(), names


def checked_losses(prediction, truth, metric, *, context):
    import torch
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(truth).all()):
        raise FloatingPointError("nonfinite prediction/truth at " + context)
    values = metric(prediction, truth)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite loss at " + context)
    return values


def fit(a, prepared, output, model, batches, validate, metadata, *, wave=False):
    """Original E500 Adam/OneCycle schedule; one invocation, one shot/seed/arm."""
    import torch
    data_id = training_identity(a, prepared)
    per_epoch = math.ceil(a.shots / min(20, a.shots))
    total = 500 * per_epoch
    max_lr = maximum_learning_rate(a.mode, a.task, a.shots)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=max_lr, betas=(0.9, 0.9), weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total,
                                                   pct_start=0.2, div_factor=1e4, final_div_factor=1e4)
    atomic_json(output / "PREREGISTRATION.json", {
        "recipe": prepared["recipe"], "planned_updates": total, "updates_per_epoch": per_epoch,
        "batch_size": min(20, a.shots), "optimizer": "Adam betas(0.9,0.9) wd1e-6",
        "schedule": "OneCycle pct_start0.2 div_factor1e4 final_div_factor1e4",
        "max_learning_rate": max_lr,
        "validation_metric": "free_rollout_mean_per_trajectory_per_step_L2RE" if wave else "full_rollout_per_channel_L2RE",
        "validation_every_epochs": 10, "data_identity": data_id, **metadata,
        "test_used_for_selection": False})
    best, updates = math.inf, 0
    path = output / "target_full_state.pt"
    for epoch in range(1, 501):
        model.train()
        loss_sum, sample_count = 0., 0
        for values, context in batches(epoch):
            atomic_json(output / "CURRENT_BATCH.json", {"epoch": epoch, "update": updates + 1, **context})
            loss = values.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}, update {updates + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}, update {updates + 1}")
            optimizer.step()
            schedule.step()
            updates += 1
            loss_sum += float(values.detach().sum())
            sample_count += values.numel()
            if a.smoke_updates and updates == a.smoke_updates:
                break
        smoke_done = bool(a.smoke_updates and updates == a.smoke_updates)
        score, validation_error = None, None
        if epoch % 10 == 0 or smoke_done:
            try:
                score = validate()
                if not math.isfinite(score):
                    raise FloatingPointError("nonfinite validation metric")
            except FloatingPointError as error:
                if wave:
                    raise
                # Final coefficient protocol rejects this candidate, then continues training.
                score, validation_error = None, str(error)
        save = (smoke_done and score is not None) or (a.selection == "fixed-final" and epoch == 500) or (
            a.selection == "original-val" and score is not None and score < best)
        if score is not None:
            best = min(best, score)
        if save:
            payload = checkpoint_payload(model, prepared, epoch, updates, score, data_id)
            payload["population"] = metadata
            save_tensor(path, payload)
        event = {"epoch": epoch, "updates": updates, "train_l2re": loss_sum / sample_count,
                 "validation_l2re": score, "validation_invalid_reason": validation_error,
                 "best_validation_l2re": best if math.isfinite(best) else None}
        finite_tree(event)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        atomic_json(output / "HEARTBEAT.json", event)
        if smoke_done:
            break
    if updates != (a.smoke_updates or total) or not path.is_file():
        raise RuntimeError("training did not finish the exact requested update budget")
    # Strict tensor roundtrip, followed by validation reconstruction in the
    # separate *-test process before any test array is opened.
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    checked_state(model, loaded["full_model_state"])
    restoration(loaded["validation_l2re"], validate(), output)
    return {"checkpoint": path.name, "checkpoint_sha256": sha256(path), "updates": updates,
            "completed_epochs": epoch, "selected_epoch": loaded["completed_epoch"], "test_accessed": False}


def fewshot(a, prepared, output, bound):
    import torch
    original = module(OOD + ".run_coefficient_job")
    data, spec, sampler, train, valid, population = ood_data(a, prepared)
    model, names = configure_ood(a, bound, spec)
    device = torch.device("cuda:0")
    def validate():
        result = coefficient_rollout(model, spec, valid)
        if [row["row"] for row in result["rows"]] != population["validation_rows"]:
            raise ValueError("validation row population/order drift")
        atomic_json(output / "VALIDATION_SCORES.json", result)
        return result["mean_l2re"]
    if a.mode == "fewshot-test":
        snapshot = load_target(a, prepared)
        if snapshot.get("population", {}).get("split") != population:
            raise ValueError("few-shot target row/address identities drift")
        checked_state(model, snapshot["full_model_state"])
        restoration(snapshot["validation_l2re"], validate(), output)
        return coefficient_test(a, prepared, output, model)
    def batches(epoch):
        sampler.set_epoch(epoch - 1)
        for batch in train:
            context = {"rows": batch["row"].tolist(), "starts": batch["start"].tolist()}
            atomic_json(output / "CURRENT_BATCH.json", {"epoch": epoch, **context})
            pred, truth = original._predict(model, "ours", batch, spec, device)
            values = checked_losses(pred, truth, lambda p, t: original.simple_relative_lp(p, t, reduction="none"),
                                     context=str(context))
            yield values, context
    return fit(a, prepared, output, model, batches, validate,
               {"split": population, "trainable_names": names})


def coefficient_test(a, prepared, output, model):
    import torch
    ev = module("hotpot._internal.zero_shot.evaluate")
    target = module(OOD + ".data").TARGETS[a.task]
    spec = replace(ev.EVALUATION_BY_KEY[target.key],
                   test_path=str(prepared["paths"][f"transfer.{a.task}.test"]))
    atomic_json(output / "TEST_DATA.json", identity(prepared["paths"][f"transfer.{a.task}.test"], a.repo))
    atomic_json(output / "TEST_ACCESS.json", {"test_used_for_selection": False,
                "scope": "COEFFICIENT_FEWSHOT_FULL_ROLLOUT"})
    loader = torch.utils.data.DataLoader(ev.OfficialRolloutDataset(a.repo, spec),
                                        batch_size=8, num_workers=a.num_workers)
    result = coefficient_rollout(model, target, loader)
    if [row["row"] for row in result["rows"]] != list(range(spec.samples)):
        raise ValueError("coefficient test population/order drift")
    atomic_json(output / "RESULT.json", result)
    return result


def rollout_scores(loader, spec, predict, output):
    """Native released mask/grid/channel loss, own autoregressive predictions."""
    import torch
    metric = module(BASE + ".objective").simple_relative_lp
    rows_out, scores = [], {f"h{h}": [] for h in spec.report_horizons}
    with torch.inference_mode():
        for rows, history, truth, mask in loader:
            ids = rows.tolist()
            atomic_json(output / "CURRENT_BATCH.json", {"rows": ids, "task": spec.key})
            history, truth, mask = history.cuda(), truth.cuda(), mask.cuda()
            predictions = []
            for step in range(spec.rollout_steps):
                prediction = predict(history, rows.cuda())
                if prediction.shape != (*history.shape[:3], 1, 4) or not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError(f"rollout output invalid at rows {ids}, step {step + 1}")
                predictions.append(prediction)
                history = torch.cat((history[..., 1:, :], prediction), dim=-2)
            joined = torch.cat(predictions, dim=-2)
            for h in spec.report_horizons:
                values = metric(joined[..., :h, :], truth[..., :h, :], mask, reduction="none").cpu().tolist()
                finite_tree(values)
                scores[f"h{h}"].extend(values)
            rows_out.extend(ids)
            atomic_json(output / "PARTIAL_SCORES.json", {"rows": rows_out, "per_trajectory": scores})
    if rows_out != list(range(spec.samples)):
        raise ValueError("official evaluation population/order drift")
    return {"rows": rows_out, "per_trajectory": scores,
            "mean": {h: sum(values) / len(values) for h, values in scores.items()},
            "metric": "released_original_grid_masked_per_channel_SimpleLpLoss",
            "samples": spec.samples, "rollout_steps": spec.rollout_steps}


def ood_evaluate(a, prepared, output, model):
    import torch
    ev = module("hotpot._internal.zero_shot.evaluate")
    target = module(OOD + ".data").TARGETS[a.task]
    spec = replace(ev.EVALUATION_BY_KEY[target.key], test_path=str(prepared["paths"][f"transfer.{a.task}.test"]))
    atomic_json(output / "TEST_DATA.json", identity(prepared["paths"][f"transfer.{a.task}.test"], a.repo))
    model = model.cuda().eval()
    def predict(history, rows):
        values, mask = ev.conditions(target, len(rows), history.device)
        return model(history, [target.program] * len(rows), values, mask)
    atomic_json(output / "TEST_ACCESS.json", {"test_used_for_selection": False,
                "historically_blind": False, "scope": "SAME_FAMILY_COEFFICIENT_" + a.mode.upper()})
    loader = torch.utils.data.DataLoader(ev.OfficialRolloutDataset(a.repo, spec), batch_size=4,
                                        shuffle=False, num_workers=a.num_workers)
    result = rollout_scores(loader, spec, predict, output)
    atomic_json(output / "RESULT.json", result)
    return result


def known_test(a, prepared, output, bound):
    import torch
    snapshot = load_target(a, prepared)
    checked_state(bound, snapshot["full_model_state"])
    ev = module(BASE + ".evaluate_e1000")
    specs = module(BASE + ".evaluation_manifest").EVALUATION
    spec = replace(next(s for s in specs if s.key == a.task),
                   test_path=str(prepared["paths"][f"transfer.{a.task}.test"]))
    atomic_json(output / "TEST_DATA.json", identity(prepared["paths"][f"transfer.{a.task}.test"], a.repo))
    sidecar = ev.CertifiedCFDBTestSidecar(prepared["paths"]["transfer.cfdb.test_sidecar"]) if a.task == "cfdbench" else None
    bound.cuda().eval()
    def predict(history, rows):
        return bound(history, torch.full_like(rows, spec.dataset_index), rows, cfdb_sidecar=sidecar)
    atomic_json(output / "TEST_ACCESS.json", {"fixed_endpoint": "E200", "test_used_for_selection": False})
    loader = torch.utils.data.DataLoader(ev.OfficialRolloutDataset(a.repo, spec), batch_size=2,
                                        shuffle=False, num_workers=a.num_workers)
    result = rollout_scores(loader, spec, predict, output)
    atomic_json(output / "RESULT.json", result)
    return result


def cfdb_evaluate(a, prepared, output, bound):
    """Reuse certified BC/PC/GC population, metrics and case-uniform aggregator."""
    import torch
    ev = module("hotpot._internal.cfdb_transfer.evaluate")
    conditions = module("hotpot._internal.cfdb_conditions.evaluate_condition_space")
    paths = prepared["paths"]
    atomic_json(output / "TEST_DATA.json", {
        key: identity(paths[key], a.repo) for key in ("transfer.cfdbench.test", "transfer.cfdb.train_sidecar",
                                                     "transfer.cfdb.test_sidecar", "transfer.cfdb.schema")})
    train_map = ev.load_map(paths["transfer.cfdb.train_sidecar"], scope="OFFICIAL_TRAIN_ONLY", expected_rows=9000)
    test_map = ev.load_map(paths["transfer.cfdb.test_sidecar"], scope="OFFICIAL_TEST_ONLY", expected_rows=1000)
    family = a.task.removeprefix("cfdb_")
    def metadata(root, key):
        path = checked_path(paths["transfer.cfdb.case_root"], str(key) + "/case.json")
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("CFDB case metadata must be an object")
        return result
    with patched(conditions, case_metadata=metadata):
        audit = ev.condition_audit(a.repo, family, train_map["entries"], test_map["entries"])
    selected = tuple(int(r["index"]) for r in test_map["entries"] if r["regime"] == family)
    atomic_json(output / "TEST_ACCESS.json", {"historically_blind": False, "test_used_for_selection": False,
                "case_indices": selected, "condition_space_audit": audit})
    dataset = ev.OfficialCFDBDataset(paths["transfer.cfdbench.test"], ev.load_schema(paths["transfer.cfdb.schema"]), selected)
    loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
    sidecar = ev.CertifiedTestSidecar(test_map)
    bound.cuda().eval()
    def step(history, indices):
        atomic_json(output / "CURRENT_BATCH.json", {"rows": indices.cpu().tolist(), "family": family})
        return bound(history, torch.full_like(indices, 5), indices, cfdb_sidecar=sidecar)
    raw = ev.evaluate_rows_indexed(a.source_key, loader, step, torch.device("cuda:0"), len(dataset))
    # Retain all row identities and both metrics before the native case aggregation.
    finite_tree(raw)
    atomic_json(output / "RAW_SCORES.json", raw)
    result = {"family": family, "condition_space_audit": audit,
              "results": ev.aggregate_arm(raw, test_map["entries"], family),
              "training_performed": False, "historically_blind": False,
              "test_used_for_selection": False, "source_key": a.source_key}
    finite_tree(result)
    atomic_json(output / "RESULT.json", result)
    return result




def wave_model(a, bound):
    import torch
    old = wave_models(a.repo)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260831)
        model = old.WaveResidualModel()
    private = old._state_digest(model, skip_shared=True)
    random_shared = old._state_digest(model.shared_analyzer)
    if a.arm == "transfer":
        checked_state(model.shared_analyzer, bound.model.shared_analyzer.state_dict())
    if old._state_digest(model, skip_shared=True) != private:
        raise ValueError("selected encoder reuse changed target private initialization")
    return model.cuda(), old.predict, {
        "architecture": "original WaveResidualModel Tiny17, same constructor for both arms",
        "private_init_sha256": private, "random_encoder_init_sha256": random_shared,
        "encoder_init_sha256": old._state_digest(model.shared_analyzer),
        "source_load_scope": "shared_analyzer only" if a.arm == "transfer" else "none",
        "initialization_seed": 20260831, "all_parameters_trainable": True,
        "physical_metadata": "unavailable; original zero values and zero masks"}


def wave_values(a, prepared, role):
    import h5py
    import numpy as np
    import torch
    manifest = json.loads(prepared["paths"]["transfer.wave.split"].read_text(encoding="utf-8"))
    target = validate_wave_split(manifest, a.task)
    if role == "train":
        source, ids = "train", target["fewshot_train_indices"]["0"][str(a.shots)]
    else:
        if role == "test" and a.mode != "wave-test":
            raise PermissionError("Wave test access requires the separate wave-test mode")
        source, ids = target[role]["source"], target[role]["indices"]
    path = prepared["paths"][f'transfer.wave.{a.task.removeprefix("Wave-")}.{source}']
    record = target["sources"][source]
    # Bind actual data via named paths, preserving the historical byte identity;
    # never trust or follow the old absolute path saved in the split manifest.
    if path.stat().st_size != record["bytes"] or sha256(path) != record["file_sha256"]:
        raise ValueError("Wave bound data bytes differ from the split manifest")
    before = path.stat()
    with h5py.File(path, "r") as f:
        data = f["data"]
        if list(data.shape) != record["shape"] or data.dtype != np.dtype("float32"):
            raise ValueError("Wave array shape/dtype drift")
        values = np.stack([np.asarray(data[i], dtype=np.float32) for i in ids])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or not np.isfinite(values).all():
        raise ValueError("Wave selected data changed or is nonfinite")
    values = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 3, 1, 2)))
    return values, [f"{a.task}/{source}:{i}" for i in ids]


def wave_error(pred, truth):
    import torch
    return torch.linalg.vector_norm((pred - truth).flatten(1), dim=1) / torch.linalg.vector_norm(
        truth.flatten(1), dim=1).clamp_min(1e-8)


def wave_rollout(model, predict, arm, values, ids, output):
    import torch
    all_errors = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(values), 20):
            atomic_json(output / "CURRENT_BATCH.json", {"wave_ids": ids[begin:begin + 20]})
            batch = values[begin:begin + 20]
            history = batch[:, :10].permute(0, 2, 3, 1)[..., None].contiguous().cuda()
            errors = []
            for t in range(10, batch.shape[1]):
                prediction = predict(model, arm, history)
                truth = batch[:, t, ..., None].contiguous().cuda()
                errors.append(checked_losses(prediction, truth, wave_error, context=f"Wave {begin} frame {t}").cpu())
                history = torch.cat((history[..., 1:, :], prediction.unsqueeze(-2)), dim=-2)
            all_errors.append(torch.stack(errors, dim=1))
    scores = torch.cat(all_errors)
    return {"ids": ids, "mean_l2re": float(scores.mean()), "first_step": float(scores[:, 0].mean()),
            "last_step": float(scores[:, -1].mean()), "per_step": scores.mean(0).tolist(),
            "per_trajectory_per_step": scores.tolist(), "trajectory_count": len(ids)}


def wave(a, prepared, output, bound):
    import numpy as np
    import torch
    model, predict, initialization = wave_model(a, bound)
    arm = "hetpot_" + a.arm
    validation, valid_ids = wave_values(a, prepared, "validation")
    def validate():
        return wave_rollout(model, predict, arm, validation, valid_ids, output)["mean_l2re"]
    if a.mode == "wave-test":
        snapshot = load_target(a, prepared)
        saved_init = snapshot.get("population", {}).get("initialization")
        if saved_init != initialization:
            raise ValueError("Wave target architecture/private initialization/source encoder drift")
        checked_state(model, snapshot["full_model_state"])
        restoration(snapshot["validation_l2re"], validate(), output)
        atomic_json(output / "TEST_ACCESS.json", {"test_used_for_selection": False,
                    "scope": "CUSTOM_TRAJECTORY_DISJOINT_WAVE_TEST", "historically_blind": False})
        test, ids = wave_values(a, prepared, "test")
        result = wave_rollout(model, predict, arm, test, ids, output)
        atomic_json(output / "RESULT.json", result)
        return result
    values, train_ids = wave_values(a, prepared, "train")
    def batches(epoch):
        rng = np.random.default_rng(20260915 + epoch)
        ordering = rng.permutation(a.shots)
        starts = rng.integers(0, values.shape[1] - 10, size=a.shots)
        batch_size = min(20, a.shots)
        for begin in range(0, a.shots, batch_size):
            rows = ordering[begin:begin + batch_size]
            context = {"ids": [train_ids[i] for i in rows], "starts": starts[rows].tolist()}
            atomic_json(output / "CURRENT_BATCH.json", {"epoch": epoch, **context})
            history = torch.stack([values[i, t:t + 10] for i, t in zip(rows, starts[rows])])
            history = history.permute(0, 2, 3, 1)[..., None].contiguous().cuda()
            truth = torch.stack([values[i, t + 10] for i, t in zip(rows, starts[rows])])[..., None].contiguous().cuda()
            errors = checked_losses(predict(model, arm, history), truth, wave_error, context=str(context))
            yield errors, context
    return fit(a, prepared, output, model, batches, validate,
               {"initialization": initialization, "train_ids": train_ids, "validation_ids": valid_ids}, wave=True)
