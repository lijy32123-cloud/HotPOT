from __future__ import annotations

from torch import nn

from hotpot._internal.encoder.model import (
    EVIDENCE_SELECTED_INTERFACE_REVISION,
    FINAL_INTERFACE_REVISION,
    FINITE_TIME_UPDATE_INTERFACE_REVISION,
    FNO_INVARIANT_INTERFACE_REVISION,
    LEGACY_INTERFACE_REVISION,
    NATIVE_ACTION_INTERFACE_REVISION,
    ModelOutput,
    SemanticFieldProgramFoundation,
    SWE_OBSERVATION_INTERFACE_REVISION,
    UNIFIED_TYPED_INTERFACE_REVISION,
)

from .config import FoundationScaleConfig
from .programs import (
    CFDBMaskedVelocityProgram,
    CFDBProgram,
    CNSProgram,
    DRDirectObservationProgram,
    DRProgram,
    DRKnownSourceResidualProgram,
    FNOInvariantTransportProgram,
    FNOProgram,
    FNOSourceAlignedSubstep4Program,
    FNOStageChartProgram,
    SWEDirectObservationProgram,
    SWEPhaseAwareProgram,
    SWEProgram,
)


class SemanticFieldProgramFoundationS90M(SemanticFieldProgramFoundation):
    """Scale representation, typed coupling, and action bandwidth together."""

    def __init__(
        self,
        config: FoundationScaleConfig | None = None,
        *,
        interface_revision: str = LEGACY_INTERFACE_REVISION,
    ) -> None:
        resolved = FoundationScaleConfig() if config is None else config
        super().__init__(resolved, interface_revision=interface_revision)
        
        # Only replace its fixed-32-channel heads with width-aware equivalents.
        self.programs = nn.ModuleDict(
            {
                "fno": (
                    FNOSourceAlignedSubstep4Program(self.toolbox, resolved)
                    if interface_revision == EVIDENCE_SELECTED_INTERFACE_REVISION
                    else (
                        FNOStageChartProgram(self.toolbox, resolved)
                        if interface_revision == NATIVE_ACTION_INTERFACE_REVISION
                        else (
                            FNOInvariantTransportProgram(self.toolbox, resolved)
                            if interface_revision == FNO_INVARIANT_INTERFACE_REVISION
                            else FNOProgram(self.toolbox, resolved)
                        )
                    )
                ),
                "dr": (
                    DRDirectObservationProgram(self.toolbox, resolved)
                    if interface_revision
                    in (
                        FINITE_TIME_UPDATE_INTERFACE_REVISION,
                        UNIFIED_TYPED_INTERFACE_REVISION,
                        EVIDENCE_SELECTED_INTERFACE_REVISION,
                    )
                    else (
                        DRKnownSourceResidualProgram(self.toolbox, resolved)
                        if interface_revision
                        in (FINAL_INTERFACE_REVISION, NATIVE_ACTION_INTERFACE_REVISION)
                        else DRProgram(self.toolbox, resolved)
                    )
                ),
                "swe": (
                    SWEPhaseAwareProgram(self.toolbox, resolved)
                    if interface_revision == NATIVE_ACTION_INTERFACE_REVISION
                    else (
                        SWEDirectObservationProgram(self.toolbox, resolved)
                        if interface_revision
                        in (
                            SWE_OBSERVATION_INTERFACE_REVISION,
                            FINITE_TIME_UPDATE_INTERFACE_REVISION,
                            UNIFIED_TYPED_INTERFACE_REVISION,
                            EVIDENCE_SELECTED_INTERFACE_REVISION,
                        )
                        else SWEProgram(self.toolbox, resolved)
                    )
                ),
                "cns": CNSProgram(self.toolbox, resolved),
                "cfdb": (
                    CFDBMaskedVelocityProgram(self.toolbox, resolved)
                    if interface_revision == UNIFIED_TYPED_INTERFACE_REVISION
                    else CFDBProgram(self.toolbox, resolved)
                ),
            }
        )


__all__ = ["ModelOutput", "SemanticFieldProgramFoundationS90M"]
