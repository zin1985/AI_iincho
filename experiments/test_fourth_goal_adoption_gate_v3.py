"""Revalidation tests for tentative strategies, not real N's preferences."""
from unittest.mock import patch
from tools import fourth_goal_adoption_gate_v3 as v3

NOW=500000.
BASE={"rest":.4,"wander":.3,"play":.3}
SUPPORTED={"strength":.27,"weights":{"rest":.2,"wander":.2,"play":.6},
           "reason":"observed_goal_support"}

def _source(passed,ages):
    return [{"action":"play","passed":bool(passed),"at":NOW-age*3600,
       "source":f"test#{i}"} for i,age in enumerate(ages)]

def test_two_separate_decline_windows_pause_bias():
    recent=[{"action":"play","passed":False,"at":NOW-i*3600.,"source":f"recent#{i}"}
            for i in range(12)]
    older=[{"action":"play","passed":True,"at":NOW-(i+12)*3600.,"source":f"older#{i}"}
            for i in range(12)]
    with patch.object(v3.v2,"gate",return_value=dict(SUPPORTED)),patch.object(v3.task,"evidence",return_value=recent+older):
        result=v3.gate(None,"sim_fourth",NOW,"scenario:test",BASE,"play")
    assert result["strength"]==0.
    assert result["weights"]==BASE
    assert result["reason"]=="confirmed_recent_goal_deterioration"

def test_stale_goal_evidence_pauses_bias():
    older=[{"action":"play","passed":True,"at":NOW-300000.,
       "source":"old#1"}]
    with patch.object(v3.v2,"gate",return_value=dict(SUPPORTED)),patch.object(v3.task,"evidence",return_value=older):
        result=v3.gate(None,"sim_fourth",NOW,"scenario:test",BASE,"play")
    assert result["reason"]=="goal_evidence_stale"
    assert result["weights"]==BASE

def test_stable_goal_is_unaffected():
    current=[{"action":"play","passed":True,"at":NOW-i*3600.,"source":f"current#{i}"}
             for i in range(24)]
    with patch.object(v3.v2,"gate",return_value=dict(SUPPORTED)),patch.object(v3.task,"evidence",return_value=current):
        result=v3.gate(None,"sim_fourth",NOW,"scenario:test",BASE,"play")
    assert result["strength"]==SUPPORTED["strength"]
    assert result["reason"]=="observed_goal_support"
