"""Safety tests: no goal rewrites, bounded trials, existing policy preserved."""
from unittest.mock import patch
from tools import fourth_goal_trial_bandit_v5 as guard
BASE={"rest":.62,"wander":.25,"play":.13}
def assess(signal):
    with patch.object(guard.pilot,"gate",return_value=signal):
        return guard.gate(None,"sim_fourth",3000.,"scenario:test",BASE,"rest")
def result(reason,alternative="wander",counts=None,rec=None):
    return {"reason":reason,"effective_goal":alternative,
            "counts":counts or {"rest":16,"wander":6,"play":4},
            "recent_estimates":rec or {"rest":.31,"wander":.72,"play":.5},
            "sources":["verified#1"]}
def test_normal_case_identical_to_proven_existing_policy():
    got=assess(result("current_goal_confirmed"))
    expected=guard.pipe.decide_probabilities(BASE,"rest",weight=.42)
    assert got["weights"]==expected
    assert not got["changed_goal"]
def test_bounded_trial_when_alternative_has_real_proven_results():
    got=assess(result("trial_alternative_after_verified_decline"))
    target=guard.pipe.decide_probabilities(BASE,"rest",weight=.42)
    assert got["trial_candidate"]=="wander"
    assert got["reason"]=="bounded_empirical_alternative_trial"
    assert got["weights"]["wander"]>target["wander"]
    assert got["strength"]==.12
    assert not got["changed_goal"]
def test_sparse_alternative_is_not_used():
    got=assess(result("trial_alternative_after_verified_decline",
        counts={"rest":16,"wander":2,"play":4}))
    assert got["reason"]=="baseline_goal_kept_weak_alternative"
    assert got["strength"]==0
def test_small_posterior_gain_does_not_experiment():
    got=assess(result("trial_alternative_after_verified_decline",
        rec={"rest":.55,"wander":.60,"play":.5}))
    assert got["strength"]==0
def test_goal_remains_original_and_probs_sum_to_one():
    got=assess(result("trial_alternative_after_verified_decline"))
    assert got["effective_goal"]=="rest"
    assert abs(sum(got["weights"].values())-1)<1e-10
    assert all(x>0 for x in got["weights"].values())
