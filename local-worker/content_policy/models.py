"""JSON-serializable result types shared by DUBVI content-policy stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


POLICY_SCHEMA_VERSION = 1


class PolicyDecision(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    REVIEW = "review"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PolicyStageResult:
    stage: str
    decision: PolicyDecision
    reason_codes: tuple[str, ...] = ()
    human_summary: str = ""
    confidence: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.decision in {
            PolicyDecision.REVIEW,
            PolicyDecision.ERROR,
            PolicyDecision.UNAVAILABLE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "human_summary": self.human_summary,
            "confidence": self.confidence,
            "metrics": dict(self.metrics),
            "details": dict(self.details),
            "needs_review": self.needs_review,
        }


@dataclass(frozen=True)
class PolicyResult:
    overall_decision: PolicyDecision
    stage_results: Sequence[PolicyStageResult]
    reason_codes: tuple[str, ...]
    human_summary: str
    needs_review: bool
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    policy_schema_version: int = POLICY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_schema_version": self.policy_schema_version,
            "overall_decision": self.overall_decision.value,
            "stage_results": [result.to_dict() for result in self.stage_results],
            "reason_codes": list(self.reason_codes),
            "human_summary": self.human_summary,
            "needs_review": self.needs_review,
            "evaluated_at": self.evaluated_at,
        }


def combine_stage_results(stage_results: Sequence[PolicyStageResult]) -> PolicyResult:
    """Apply stable policy precedence without allowing later stages to erase risk.

    Precedence is REJECT > ERROR > REVIEW > UNAVAILABLE > PASS. An unavailable
    optional stage remains visible and becomes REVIEW when paired with otherwise
    passing stages; it never silently upgrades the overall result to PASS.
    """
    stages = tuple(stage_results)
    if not stages:
        return PolicyResult(
            overall_decision=PolicyDecision.REVIEW,
            stage_results=stages,
            reason_codes=("NO_POLICY_STAGES",),
            human_summary="No policy stages were evaluated.",
            needs_review=True,
        )

    decisions = {stage.decision for stage in stages}
    if PolicyDecision.REJECT in decisions:
        overall = PolicyDecision.REJECT
    elif PolicyDecision.ERROR in decisions:
        overall = PolicyDecision.ERROR
    elif PolicyDecision.REVIEW in decisions:
        overall = PolicyDecision.REVIEW
    elif PolicyDecision.UNAVAILABLE in decisions:
        overall = PolicyDecision.UNAVAILABLE if decisions == {PolicyDecision.UNAVAILABLE} else PolicyDecision.REVIEW
    else:
        overall = PolicyDecision.PASS

    reason_codes = tuple(dict.fromkeys(code for stage in stages for code in stage.reason_codes))
    summaries = [stage.human_summary for stage in stages if stage.human_summary]
    return PolicyResult(
        overall_decision=overall,
        stage_results=stages,
        reason_codes=reason_codes,
        human_summary=" ".join(summaries),
        needs_review=overall in {
            PolicyDecision.REVIEW,
            PolicyDecision.ERROR,
            PolicyDecision.UNAVAILABLE,
        },
    )
