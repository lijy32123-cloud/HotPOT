from __future__ import annotations


from collections import defaultdict


import json


from pathlib import Path


import numpy as np


HORIZONS = (1, 5, 10)


SUBSETS = ("bc", "prop", "geo")


SUBSET_LABELS = {"bc": "BC", "prop": "PC", "geo": "GC"}


CONDITION_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "cavity": {
        "bc": ("vel_top",),
        "prop": ("density", "viscosity"),
        "geo": ("width", "height", "rotated"),
    },
    "cylinder": {
        "bc": ("vel_in",),
        "prop": ("density", "viscosity"),
        "geo": ("radius", "x_min", "x_max", "y_min", "y_max"),
    },
    "tube": {
        "bc": ("vel_in",),
        "prop": ("density", "viscosity"),
        "geo": ("width", "height"),
    },
}


def load_map(path: Path, *, scope: str, expected_rows: int) -> dict[str, object]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if (
        payload.get("mapping_certified") is not True
        or payload.get("scope") != scope
        or not isinstance(entries, list)
        or len(entries) != expected_rows
        or [int(item["index"]) for item in entries] != list(range(expected_rows))
    ):
        raise ValueError(f"certified context-map contract drifted: {path}")
    return payload


def canonical(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 12)
    raise TypeError(f"unsupported condition value {value!r}")


def case_metadata(root: Path, case_key: str) -> dict[str, object]:
    path = (root / "data/large/cfdbench" / case_key / "case.json").resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid case metadata {path}")
    return payload


def condition_audit(
    root: Path,
    family: str,
    train_entries: list[dict[str, object]],
    test_entries: list[dict[str, object]],
) -> dict[str, object]:
    train_cases = {str(item["case_key"]) for item in train_entries}
    test_cases = {str(item["case_key"]) for item in test_entries}
    overlap = train_cases & test_cases
    if overlap:
        raise RuntimeError(f"train/test case-key overlap detected: {sorted(overlap)[:4]}")

    output: dict[str, object] = {}
    for subset in SUBSETS:
        fields = CONDITION_FIELDS[family][subset]
        prefix = f"{family}/{subset}/"
        train_keys = sorted(key for key in train_cases if key.startswith(prefix))
        test_keys = sorted(key for key in test_cases if key.startswith(prefix))
        if not train_keys or not test_keys:
            raise RuntimeError(f"{family}/{subset} lacks train or test cases")
        train_meta = {key: case_metadata(root, key) for key in train_keys}
        test_meta = {key: case_metadata(root, key) for key in test_keys}
        for key, metadata in (*train_meta.items(), *test_meta.items()):
            missing = set(fields) - set(metadata)
            if missing:
                raise ValueError(f"{key} lacks condition fields {sorted(missing)}")

        train_tuples = {
            tuple(canonical(train_meta[key][field]) for field in fields)
            for key in train_keys
        }
        train_values = {
            field: tuple(canonical(train_meta[key][field]) for key in train_keys)
            for field in fields
        }
        cases: dict[str, object] = {}
        class_counts: dict[str, int] = defaultdict(int)
        for key in test_keys:
            metadata = test_meta[key]
            values = tuple(canonical(metadata[field]) for field in fields)
            field_status: dict[str, str] = {}
            all_values_seen = True
            all_inside_support = True
            for field, value in zip(fields, values):
                reference = train_values[field]
                if value in reference:
                    status = "seen_value"
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric = [float(item) for item in reference]
                    if min(numeric) <= float(value) <= max(numeric):
                        status = "interpolation_within_train_range"
                    else:
                        status = "extrapolation_outside_train_range"
                        all_inside_support = False
                    all_values_seen = False
                else:
                    status = "unseen_categorical_support"
                    all_values_seen = False
                    all_inside_support = False
                field_status[field] = status
            if values in train_tuples:
                classification = "seen_condition_tuple_new_case"
            elif all_values_seen:
                classification = "novel_combination_of_seen_values"
            elif all_inside_support:
                classification = "interpolation_within_train_ranges"
            else:
                classification = "extrapolation_outside_train_support"
            class_counts[classification] += 1
            derived: dict[str, float] = {}
            if "density" in metadata and "viscosity" in metadata:
                density = float(metadata["density"])
                if density > 0:
                    derived["kinematic_viscosity"] = float(metadata["viscosity"]) / density
            cases[key] = {
                "condition": {field: metadata[field] for field in fields},
                "derived": derived,
                "field_status": field_status,
                "classification": classification,
            }
        output[subset] = {
            "paper_label": SUBSET_LABELS[subset],
            "condition_fields": list(fields),
            "train_cases": len(train_keys),
            "test_cases": len(test_keys),
            "train_condition_tuples": len(train_tuples),
            "classification_counts": dict(sorted(class_counts.items())),
            "cases": cases,
        }
    return output


def aggregate_arm(
    scores: dict[str, dict[int, dict[int, float]]],
    entries: list[dict[str, object]],
    family: str,
) -> dict[str, object]:
    entry_by_index = {int(item["index"]): item for item in entries}
    source_indices = sorted(scores["official_masked"][1])

    def summarize(indices: list[int]) -> dict[str, object]:
        case_rows: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            case_rows[str(entry_by_index[index]["case_key"])].append(index)
        metrics: dict[str, object] = {}
        per_case: dict[str, object] = {key: {"rows": len(rows)} for key, rows in case_rows.items()}
        for metric_name in ("official_masked", "legacy_dense"):
            row_weighted: dict[str, float] = {}
            case_uniform: dict[str, float] = {}
            for horizon in HORIZONS:
                row_values = [scores[metric_name][horizon][index] for index in indices]
                case_values = []
                for key, rows in case_rows.items():
                    value = float(np.mean([scores[metric_name][horizon][index] for index in rows]))
                    per_case[key].setdefault(metric_name, {})[f"h{horizon}"] = value
                    case_values.append(value)
                row_weighted[f"h{horizon}"] = float(np.mean(row_values))
                case_uniform[f"h{horizon}"] = float(np.mean(case_values))
            metrics[metric_name] = {
                "row_weighted": row_weighted,
                "case_uniform": case_uniform,
            }
        return {
            "rows": len(indices),
            "cases": len(case_rows),
            "metrics": metrics,
            "per_case": dict(sorted(per_case.items())),
        }

    by_subset: dict[str, object] = {}
    for subset in SUBSETS:
        indices = [
            index
            for index in source_indices
            if str(entry_by_index[index]["regime"]) == family
            and str(entry_by_index[index]["subset"]) == subset
        ]
        if not indices:
            raise RuntimeError(f"{family}/{subset} has no evaluated rows")
        by_subset[subset] = summarize(indices)
    return {
        "family": summarize(source_indices),
        "by_subset": by_subset,
        "cuda_peak_allocated_gib": float(scores["runtime"][0][0]),
    }
