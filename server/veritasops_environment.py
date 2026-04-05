#server/veritasops_environment.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Veritasops Environment Implementation.

A misinformation-response simulation environment for OpenEnv.
Perfect for testing HTTP server infrastructure.
"""

from copy import deepcopy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        VeritasopsAction,
        VeritasopsObservation,
        ClaimState,
        ResourceState,
        SpreadState,
        EvidenceItem,
    )
    from ..tasks import load_task
    from ..simulator import apply_spread_dynamics, reduce_uncertainty, add_extra_evidence
    from ..rewards import compute_step_reward
    from ..utils import find_claim
    from ..grader import grade_episode
except ImportError:
    from models import (
        VeritasopsAction,
        VeritasopsObservation,
        ClaimState,
        ResourceState,
        SpreadState,
        EvidenceItem,
    )
    from tasks import load_task
    from simulator import apply_spread_dynamics, reduce_uncertainty, add_extra_evidence
    from rewards import compute_step_reward
    from utils import find_claim
    from grader import grade_episode


class VeritasopsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "task_easy"):
        self.task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._data = {}
        self._done = False

    def reset(self) -> VeritasopsObservation:
        task = load_task(self.task_id)
        claims = deepcopy(task["claims"])

        for claim in claims:
            claim["status"] = "unresolved"
            claim["warning_label_active"] = False
            claim["spread_limited"] = False
            claim["priority"] = "medium"
            claim["verification_count"] = 0
            claim["evidence_requests"] = 0

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._data = {
            "task_id": task["task_id"],
            "time_step": 0,
            "max_steps": task["max_steps"],
            "claims": claims,
            "resources": deepcopy(task["resources"]),
            "initial_resources": deepcopy(task["resources"]),
            "incoming_reports": [],
            "last_action_result": "Environment reset",
            "action_history": [],
        }
        self._done = False
        return self._build_observation()

    def step(self, action: VeritasopsAction) -> VeritasopsObservation:  # type: ignore[override]
        if "claims" not in self._data:
            self.reset()

        if self._done:
            return self._build_observation(final_score=grade_episode(self._data))

        reward = 0.0
        valid = True
        repeated = False
        premature_finalize = False

        unresolved_claims = [c for c in self._data["claims"] if c["status"] == "unresolved"]

        # Finalize strategy handling
        if action.action_type == "finalize_strategy":
            if len(unresolved_claims) > 0 and self._data["time_step"] < self._data["max_steps"] - 1:
                premature_finalize = True
                reward = compute_step_reward(
                    action_type="finalize_strategy",
                    claim=None,
                    valid=True,
                    repeated=False,
                    premature_finalize=True,
                )
                self._data["last_action_result"] = "Strategy finalized too early"
            else:
                self._data["last_action_result"] = "Strategy finalized"

            self._done = True
            return self._build_observation(
                final_score=grade_episode(self._data),
                reward=reward
            )

        claim = find_claim(self._data["claims"], action.claim_id) if action.claim_id else None

        if not claim:
            valid = False
            reward = -0.5
            self._data["last_action_result"] = "Invalid claim_id"

        else:
            # Repeated / exploit handling
            if claim["status"] != "unresolved" and action.action_type in {
                "verify_claim",
                "request_more_evidence",
                "mark_supported",
                "mark_false",
                "mark_uncertain",
            }:
                valid = False
                self._data["last_action_result"] = f"Claim {claim['claim_id']} already resolved"

            elif action.action_type == "add_warning_label" and claim.get("warning_label_active"):
                repeated = True
                self._data["last_action_result"] = f"Warning label already active for {claim['claim_id']}"

            elif action.action_type == "limit_spread" and claim.get("spread_limited"):
                repeated = True
                self._data["last_action_result"] = f"Spread already limited for {claim['claim_id']}"

            elif action.action_type == "verify_claim" and claim.get("verification_count", 0) >= 2:
                repeated = True
                self._data["last_action_result"] = f"Too many verifications for {claim['claim_id']}"

            elif action.action_type == "request_more_evidence" and claim.get("evidence_requests", 0) >= 2:
                repeated = True
                self._data["last_action_result"] = f"Too many evidence requests for {claim['claim_id']}"

            else:
                if action.action_type == "verify_claim":
                    if self._data["resources"]["verification_budget"] > 0:
                        self._data["resources"]["verification_budget"] -= 1
                        claim["verification_count"] += 1
                        reduce_uncertainty(claim, 0.25)
                        self._data["last_action_result"] = f"Verified {claim['claim_id']}"
                    else:
                        valid = False
                        self._data["last_action_result"] = "No verification budget left"

                elif action.action_type == "request_more_evidence":
                    if self._data["resources"]["verification_budget"] > 0:
                        self._data["resources"]["verification_budget"] -= 1
                        claim["evidence_requests"] += 1
                        add_extra_evidence(claim)
                        self._data["last_action_result"] = f"Requested more evidence for {claim['claim_id']}"
                    else:
                        valid = False
                        self._data["last_action_result"] = "No verification budget left"

                elif action.action_type == "mark_supported":
                    claim["status"] = "supported"
                    self._data["last_action_result"] = f"Marked {claim['claim_id']} as supported"

                elif action.action_type == "mark_false":
                    claim["status"] = "false"
                    self._data["last_action_result"] = f"Marked {claim['claim_id']} as false"

                elif action.action_type == "mark_uncertain":
                    claim["status"] = "uncertain"
                    self._data["last_action_result"] = f"Marked {claim['claim_id']} as uncertain"

                elif action.action_type == "add_warning_label":
                    if self._data["resources"]["intervention_budget"] > 0:
                        self._data["resources"]["intervention_budget"] -= 1
                        claim["warning_label_active"] = True
                        self._data["last_action_result"] = f"Added warning label to {claim['claim_id']}"
                    else:
                        valid = False
                        self._data["last_action_result"] = "No intervention budget left"

                elif action.action_type == "limit_spread":
                    if self._data["resources"]["intervention_budget"] > 0:
                        self._data["resources"]["intervention_budget"] -= 1
                        claim["spread_limited"] = True
                        self._data["last_action_result"] = f"Limited spread of {claim['claim_id']}"
                    else:
                        valid = False
                        self._data["last_action_result"] = "No intervention budget left"

                elif action.action_type == "broadcast_alert":
                    if self._data["resources"]["intervention_budget"] > 0:
                        self._data["resources"]["intervention_budget"] -= 1
                        claim["warning_label_active"] = True
                        claim["spread_limited"] = True
                        self._data["last_action_result"] = f"Broadcast alert for {claim['claim_id']}"
                    else:
                        valid = False
                        self._data["last_action_result"] = "No intervention budget left"

                elif action.action_type == "set_priority":
                    if action.priority_level:
                        claim["priority"] = action.priority_level
                        self._data["last_action_result"] = (
                            f"Set priority of {claim['claim_id']} to {action.priority_level}"
                        )
                    else:
                        valid = False
                        self._data["last_action_result"] = "Priority level missing"

            reward = compute_step_reward(
                action.action_type,
                claim,
                valid=valid,
                repeated=repeated,
                premature_finalize=False,
            )

        for c in self._data["claims"]:
            if c["status"] == "unresolved":
                apply_spread_dynamics(c)

        self._data["action_history"].append(action.model_dump())
        self._data["time_step"] += 1
        self._state.step_count += 1

        if self._data["time_step"] >= self._data["max_steps"]:
            self._done = True
            return self._build_observation(
                final_score=grade_episode(self._data),
                reward=reward
            )

        return self._build_observation(reward=reward)

    def _build_observation(self, final_score=None, reward=0.0) -> VeritasopsObservation:
        claims_obs = []
        for c in self._data["claims"]:
            claims_obs.append(
                ClaimState(
                    claim_id=c["claim_id"],
                    text=c["text"],
                    evidence=[EvidenceItem(**e) for e in c["evidence"]],
                    uncertainty=c["uncertainty"],
                    risk_level=c["risk_level"],
                    spread=SpreadState(**c["spread"]),
                    status=c["status"],
                    warning_label_active=c["warning_label_active"],
                    spread_limited=c["spread_limited"],
                    priority=c["priority"],
                )
            )

        return VeritasopsObservation(
            time_step=self._data["time_step"],
            max_steps=self._data["max_steps"],
            active_claims=claims_obs,
            resources=ResourceState(**self._data["resources"]),
            incoming_reports=self._data["incoming_reports"],
            last_action_result=self._data["last_action_result"],
            remaining_steps=self._data["max_steps"] - self._data["time_step"],
            final_score=final_score,
            metadata={
                "reward": reward,
                "done": self._done,
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def debug_state(self) -> dict:
        """
        Full internal state for debugging / local evaluation.
        Safe to use in tests and offline benchmarking.
        """
        return deepcopy(self._data)