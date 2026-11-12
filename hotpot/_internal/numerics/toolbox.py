from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from torch import Tensor, nn


class _AuditedOperator(nn.Module):
    def __init__(self, identifier: str) -> None:
        super().__init__()
        self.identifier = identifier
        self.owner_calls: Counter[str] = Counter()

    def record(self, owner: str) -> None:
        if not owner:
            raise ValueError("toolbox owner must be non-empty")
        self.owner_calls[owner] += 1

    def reset_audit(self) -> None:
        self.owner_calls.clear()


class AddCompatibleTerms(_AuditedOperator):
    def forward(self, left: Tensor, right: Tensor, *, owner: str) -> Tensor:
        self.record(owner)
        if left.shape != right.shape:
            raise ValueError("compatible terms must have identical shapes")
        return left + right


class AffineStateUpdate(_AuditedOperator):
    def forward(
        self,
        base: Tensor,
        increment: Tensor,
        coefficient: Tensor | float,
        *,
        owner: str,
    ) -> Tensor:
        self.record(owner)
        if base.shape != increment.shape:
            raise ValueError("base and increment must have identical shapes")
        return base + coefficient * increment


class ExplicitMidpoint(_AuditedOperator):
    def forward(
        self,
        base: Tensor,
        step: float,
        rhs: Callable[[Tensor], Tensor],
        affine: AffineStateUpdate,
        *,
        owner: str,
    ) -> Tensor:
        self.record(owner)
        first = rhs(base)
        midpoint = affine(base, first, 0.5 * step, owner=owner)
        second = rhs(midpoint)
        return affine(base, second, step, owner=owner)


class StrangSplit(_AuditedOperator):
    def forward(
        self,
        state: Tensor,
        half_flow: Callable[[Tensor], Tensor],
        nonlinear_flow: Callable[[Tensor], Tensor],
        *,
        owner: str,
    ) -> Tensor:
        self.record(owner)
        state = half_flow(state)
        state = nonlinear_flow(state)
        return half_flow(state)


class UniformMeanZero(_AuditedOperator):
    def forward(self, value: Tensor, *, owner: str) -> Tensor:
        self.record(owner)
        return value - value.mean(dim=(-2, -1), keepdim=True)


class RestoreUniformMean(_AuditedOperator):
    def forward(
        self, reference: Tensor, candidate: Tensor, *, owner: str
    ) -> Tensor:
        self.record(owner)
        if reference.shape != candidate.shape:
            raise ValueError("reference and candidate must have identical shapes")
        return (
            candidate
            + reference.mean(dim=(-2, -1), keepdim=True)
            - candidate.mean(dim=(-2, -1), keepdim=True)
        )


class CanonicalNumericalToolbox(nn.Module):
    """Canonical owner of every exactly shared parameter-free implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.add_terms = AddCompatibleTerms("algebra.add_compatible_terms")
        self.affine_update = AffineStateUpdate("algebra.affine_state_update")
        self.explicit_midpoint = ExplicitMidpoint("time.explicit_midpoint")
        self.strang_split = StrangSplit("time.strang_split")
        self.uniform_mean_zero = UniformMeanZero("constraint.uniform_mean_zero")
        self.restore_uniform_mean = RestoreUniformMean(
            "constraint.restore_uniform_mean"
        )

    def reset_audit(self) -> None:
        for operator in self.children():
            if not isinstance(operator, _AuditedOperator):
                raise TypeError("toolbox contains a non-audited child")
            operator.reset_audit()

    def audit_payload(self) -> dict[str, object]:
        operators: dict[str, object] = {}
        for operator in self.children():
            if not isinstance(operator, _AuditedOperator):
                raise TypeError("toolbox contains a non-audited child")
            operators[operator.identifier] = {
                "runtime_object_id": id(operator),
                "owner_calls": dict(sorted(operator.owner_calls.items())),
                "parameter_count": sum(value.numel() for value in operator.parameters()),
            }
        return {
            "schema": "moepot_v3_toolbox.executable_shared_algorithms.v3",
            "operator_count": len(operators),
            "trainable_parameters": sum(value.numel() for value in self.parameters()),
            "operators": operators,
        }


__all__ = ["CanonicalNumericalToolbox"]
