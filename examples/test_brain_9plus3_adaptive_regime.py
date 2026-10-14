"""Evidence-driven adaptation: no oracle, no invented emotional state."""
from unittest.mock import patch
import brain_9plus3_adaptive_regime as adaptive

def _events(old,mid,new,action="play"):
    # chronologically oldest => newest then reverse, as SQL evidence does
    b=[{"action":action,"passed":bool(x),"at":i*3600.,
        "source":f"brain_9plus3_task_outcomes#{i}"}
       for i,x in enumerate(list(old)+list(mid)+list(new))]
    return list(reversed(b))
def _check(events):
    with patch.object(adaptive,"evidence",return_value=events):
        return adaptive.advise(None,"sim_fourth",200000.,"scenario:material_observation")

def test_no_early_alarm():
    r=_check(_events([True]*6,[False]*6,[False]*6))
    assert r["changed"] is False

def test_one_bad_window_not_enough():
    r=_check(_events([True]*18,[True]*6,[False]*6))
    assert r["changed"] is False

def test_two_disjoint_bad_windows_detect():
    r=_check(_events([True]*18,[False]*6,[False]*6))
    assert r["changed"] is True
    assert r["half_life_days"]==1.5
    assert r["epsilon"]==.32
    assert len(r["sources"])==30
    assert r["subjective_feelings_inferred"] is False

def test_stable_results_keep_original_exploration():
    r=_check(_events([True]*18,[True]*6,[True]*6))
    assert r["changed"] is False
    assert r["epsilon"]==.22
    assert r["half_life_days"]==0.

def test_changes_in_action_mix_not_confused_with_drift():
    old=_events([True]*18,[False]*6,[False]*6)
    for i,item in enumerate(old[:12]):
        item["action"]="rest"
    r=_check(old)
    assert r["changed"] is False

def test_rebalance_changes_action_preferences_using_only_observed_rewards():
    import brain_9plus3_task_learning as task
    observed=_events([True]*18,[False]*6,[False]*6)
    with patch.object(task,"evidence",return_value=observed):
        prior={"rest":.1,"wander":.08,"play":.82}
        shifted=adaptive.rebalance(None,"sim_fourth",200000.,
                   "scenario:material_observation",prior)
    assert abs(sum(shifted.values())-1.)<1e-8
    assert shifted["play"]<prior["play"]
    assert shifted["rest"]>prior["rest"]
    assert all(v>0 for v in shifted.values())

def test_confirmed_failing_incumbent_is_temporarily_deemphasized():
    import brain_9plus3_task_learning as task
    observed=_events([True]*18,[False]*6,[False]*6)
    with patch.object(task,"evidence",return_value=observed):
        prior={"rest":.1,"wander":.08,"play":.82}
        result=adaptive.rebalance(None,"sim_fourth",200000.,
           "scenario:material_observation",prior,declining_action="play")
    assert result["play"]<.35
    assert result["rest"]+result["wander"]>.65
    assert abs(sum(result.values())-1.)<1e-8