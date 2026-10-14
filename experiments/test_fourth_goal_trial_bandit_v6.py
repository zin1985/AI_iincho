"""Bounded virtual-world alternative trials, never real N actions."""
from unittest.mock import patch
import sqlite3
import pytest
from tools import fourth_goal_trial_bandit_v6 as v6
BASE={"rest":.6,"wander":.3,"play":.1}
GOOD={"reason":"trial_alternative_after_verified_decline",
      "effective_goal":"wander","counts":{"rest":24,"wander":8,"play":2},
      "recent_estimates":{"rest":.25,"wander":.64,"play":.5},
      "sources":["brain_9plus3_task_outcomes#123"]}
def make():
    return sqlite3.connect(":memory:")
def test_only_fourth_persona_can_request_trials():
    with make() as c:
        with pytest.raises(ValueError):
            v6.gate(c,"iincho_n",100000.,"scenario:test",BASE,"rest")
def test_no_observed_decline_means_existing_policy_unchanged():
    with make() as c:
        with patch.object(v6.detector,"gate",return_value={**GOOD,"reason":"current_goal_confirmed"}):
            r=v6.gate(c,"sim_fourth",100000.,"scenario:test",BASE,"rest")
        assert r["weights"]==v6.original.decide_probabilities(BASE,"rest",weight=.42)
        assert r["trial_action_requested"] is None
        assert not c.execute("select name from sqlite_master where name='brain_9plus3_trial_intents'").fetchone()
def test_one_per_day_and_never_claim_outcome():
    with make() as c:
        with patch.object(v6.detector,"gate",return_value=GOOD):
            a=v6.gate(c,"sim_fourth",100000.,"scenario:test",BASE,"rest")
            b=v6.gate(c,"sim_fourth",100000.+3600,"scenario:test",BASE,"rest")
            d=v6.gate(c,"sim_fourth",100000.+86400,"scenario:test",BASE,"rest")
        assert a["trial_action_requested"]=="wander"
        assert a["weights"]["wander"]==1.0
        assert b["trial_action_requested"] is None
        assert d["trial_action_requested"]=="wander"
        assert d["objective_reward_not_inferred"]
        assert c.execute("select count(*) from brain_9plus3_trial_intents").fetchone()[0]==2
def test_sparse_alternative_does_not_schedule():
    with make() as c:
        with patch.object(v6.detector,"gate",return_value={
             **GOOD,"counts":{"rest":24,"wander":2,"play":2}}):
            a=v6.gate(c,"sim_fourth",100000.,"scenario:test",BASE,"rest")
        assert a["trial_action_requested"] is None
def test_context_separation_for_trial_budget():
    with make() as c:
        with patch.object(v6.detector,"gate",return_value=GOOD):
            a=v6.gate(c,"sim_fourth",100000.,"scenario:a",BASE,"rest")
            b=v6.gate(c,"sim_fourth",100000.,"scenario:b",BASE,"rest")
        assert a["trial_action_requested"]==b["trial_action_requested"]=="wander"
        assert c.execute("select count(*) from brain_9plus3_trial_intents").fetchone()[0]==2
