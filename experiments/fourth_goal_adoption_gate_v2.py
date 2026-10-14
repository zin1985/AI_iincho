"""Evidence-gated influence of a synthetic 9+3 goal on chosen actions.

The three integrators can propose a new *test strategy*. This gate checks that
observed task outcomes justify changing the action distribution. A goal that
has little or weak supporting evidence remains tentative and cannot override
a well-supported task policy. No hidden oracle, unchosen results or N state.
"""
from __future__ import annotations
import math
import brain_9plus3_task_learning as tasks

ACTIONS=("rest","wander","play")
def _unit(prob):
    positive={k:max(0.,float(v)) for k,v in prob.items()}
    total=sum(positive.values())
    if total<=0:raise ValueError("invalid action distribution")
    return {k:v/total for k,v in positive.items()}

def gate(c,profile,now,context,prob,goal,*,max_weight=.42):
    original=_unit(prob)
    if goal not in original:return {"weights":original,"strength":0.,"reason":"no_goal",
       "observations":0,"goal_estimate":None,"base_estimate":None}
    events=tasks.evidence(c,profile,now,context,limit=128)
    post=tasks.posterior(events,list(original),now=now,half_life_days=4.)
    estimates=post["expected_task_success"]
    counts=post["counts"]
    n=sum(counts.values())
    effective=float(post["effective_observations"])
    goal_n=counts.get(goal,0)
    if n<8 or goal_n<3:
        strength=0.
        reason="insufficient_observed_goal_evidence"
    else:
        baseline_estimate=sum(original[a]*estimates[a] for a in original)
        gap=estimates[goal]-baseline_estimate
        effective_ratio=min(1.,effective/(effective+10.))
        # The goal is a hypothesis and needs empirical support. A negative
        # gap must not turn the objective-learning policy into a weaker one.
        if gap<-.035:
            strength=.0
            reason="goal_underperforms_existing_policy"
        elif gap<.015:
            strength=.08*effective_ratio
            reason="goal_advantage_unproven"
        else:
            strength=max_weight*min(1.,goal_n/(goal_n+6.))*(.65+.35*effective_ratio)
            reason="observed_goal_support"
    # Keep every candidate possible. Intended support for downstream choice.
    target={a:(.82 if a==goal else .09) for a in original}
    updated=_unit({a:(1-strength)*original[a]+strength*target[a] for a in original})
    return {"weights":updated,"strength":round(strength,5),
            "reason":reason,"observations":n,
            "goal_observations":goal_n,
            "goal_estimate":round(estimates[goal],5),
            "base_estimate":round(sum(original[a]*estimates[a]
                                      for a in original),5),
            "source_ids":[x["source"] for x in events[:8]],
            "not_a_personal_emotion":True}
