"""Experiment-only verified-outcome gate tests. All rewards are explicit fixtures."""
from unittest.mock import patch
from tools import fourth_goal_adoption_gate_v2 as gate

BASE={"rest":.40,"wander":.30,"play":.30}
NOW=190000.

def events(rates):
    result=[]
    idx=0
    for action,(successes,failures) in rates.items():
        for outcome in ([True]*successes+[False]*failures):
            idx+=1
            result.append({"action":action,"passed":outcome,
              "at":NOW-idx*200.,"source":f"verified_fixture#{idx}"})
    return result

def check(goal,history):
    with patch.object(gate.tasks,"evidence",return_value=history):
        return gate.gate(None,"sim_fourth",NOW,"scenario:material_observation",
                         BASE,goal)

def test_never_bias_without_graded_experience():
    r=check("play",[])
    assert r["strength"]==0.
    assert r["weights"]==BASE

def test_unsupported_goal_does_not_overwrite_verified_preference():
    history=events({"rest":(24,2),"play":(1,25),"wander":(10,10)})
    r=check("play",history)
    assert r["reason"]=="goal_underperforms_existing_policy"
    assert r["strength"]==0.
    assert r["weights"]==BASE

def test_superior_goal_increases_real_action_probability():
    history=events({"rest":(2,24),"play":(25,2),"wander":(10,10)})
    r=check("play",history)
    assert r["reason"]=="observed_goal_support"
    assert r["strength"]>0.
    assert r["weights"]["play"]>BASE["play"]
    assert abs(sum(r["weights"].values())-1)<1e-9

def test_scant_goal_evidence_remains_tentative():
    history=events({"rest":(8,5),"play":(2,0),"wander":(7,6)})
    r=check("play",history)
    assert r["reason"]=="insufficient_observed_goal_evidence"
    assert r["strength"]==0.

def test_no_hidden_counterfactuals_or_memory_mutation():
    history=events({"rest":(15,15),"play":(18,9),"wander":(15,15)})
    before=[dict(x) for x in history]
    r=check("play",history)
    assert history==before
    assert r["source_ids"] and all(s.startswith("verified_fixture#") for s in r["source_ids"])
    assert r["not_a_personal_emotion"] is True
