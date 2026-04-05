# models.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Literal, Dict, Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EvidenceItem(Observation):
    text: str = Field(..., description="Evidence snippet")
    credibility: float = Field(..., ge=0.0, le=1.0, description="Credibility score")


class SpreadState(Observation):
    views: int = Field(..., description="Current number of views")
    shares: int = Field(..., description="Current number of shares")
    growth_rate: float = Field(..., description="Growth multiplier per step")


class ClaimState(Observation):
    claim_id: str = Field(..., description="Unique claim ID")
    text: str = Field(..., description="Claim text")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    risk_level: float = Field(..., ge=0.0, le=1.0)
    spread: SpreadState
    status: Literal["unresolved", "supported", "false", "uncertain"] = "unresolved"
    warning_label_active: bool = False
    spread_limited: bool = False
    priority: Literal["low", "medium", "high"] = "medium"


class ResourceState(Observation):
    verification_budget: int = Field(..., ge=0)
    intervention_budget: int = Field(..., ge=0)


class VeritasopsAction(Action):
    action_type: Literal[
        "verify_claim",
        "request_more_evidence",
        "mark_supported",
        "mark_false",
        "mark_uncertain",
        "add_warning_label",
        "limit_spread",
        "broadcast_alert",
        "set_priority",
        "finalize_strategy",
    ] = Field(..., description="Type of action to execute")

    claim_id: Optional[str] = Field(default=None, description="Target claim ID")
    priority_level: Optional[Literal["low", "medium", "high"]] = Field(
        default=None, description="Priority level if action_type=set_priority"
    )


class VeritasopsObservation(Observation):
    time_step: int = Field(..., description="Current timestep")
    max_steps: int = Field(..., description="Maximum timesteps in the episode")
    active_claims: List[ClaimState] = Field(default_factory=list)
    resources: ResourceState
    incoming_reports: List[str] = Field(default_factory=list)
    last_action_result: str = Field(default="", description="Last action summary")
    remaining_steps: int = Field(..., description="Steps left")
    final_score: Optional[float] = Field(default=None, description="Final episode score if done")
    metadata: Dict[str, Any] = Field(default_factory=dict)