"""Conservative v5 policy: preserve the existing nine-three goal when stable.

Only allocate <=12% of the next action's probability to an empirically
plausible replacement after corroborated deterioration. This is a
short *trial*, not authorization to rewrite a persona goal or memory.
"""
from __future__ import annotations
from tools import fourth_goal_trial_bandit_v4 as pilot
from tools import fourth_lived_pipeline as pipe
from tools.fourth_goal_adoption_gate_v2 import _unit
import brain_9plus3_task_learning as task

TRIAL_FRACTION=.12
def gate(c,profile,now,context,prob,goal,*,max_weight=.42):
    original=_unit(prob)
    stable=pipe.decide_probabilities(original,goal,weight=max_weight)
    signal=pilot.gate(c,profile,now,context,original,goal,max_weight=max_weight)
    candidate=signal["effective_goal"]
    counts=signal.get("counts",{})
    estimates=signal.get("recent_estimates",{})
    if (signal["reason"]=="trial_alternative_after_verified_decline" and
        candidate!=goal and counts.get(candidate,0)>=5 and
        estimates.get(candidate,0.)-estimates.get(goal,0.)>=.15):
        target={a:(.82 if a==candidate else .09) for a in stable}
        updated=_unit({a:(1.-TRIAL_FRACTION)*stable[a]+
                         TRIAL_FRACTION*target[a] for a in stable})
        reason="bounded_empirical_alternative_trial"
        fraction=TRIAL_FRACTION
    else:
        updated=stable
        reason=("baseline_goal_kept_weak_alternative" if
          signal["reason"].startswith("trial_alternative") else
          "baseline_goal_kept")
        fraction=0.
    return {"weights":updated,"strength":fraction,
       "reason":reason,"effective_goal":goal,"previous_goal":goal,
       "changed_goal":False,"sources":signal.get("sources",[]),
       "source_ids":signal.get("sources",[]),
       "observations":sum(counts.values()),
       "trial_candidate":candidate if fraction else None,
       "trial_bound":TRIAL_FRACTION,
       "baseline_policy_unmodified_when_not_trailing":True,
       "subjective_claims":False}
