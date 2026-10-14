"""Non-stationary contextual goal trials for isolated fourth-persona 9+3.

The three 9+3 integrators keep proposing a test goal. A small independent
bandit head checks *chosen-and-graded* outcomes and tests recent alternatives
when the chosen goal has deteriorated. The controller never reads the oracle,
nonselected rewards, N memories, subjective feelings or real personality data.
"""
from __future__ import annotations
import math
from tools import fourth_goal_adoption_gate_v2 as earlier
import brain_9plus3_task_learning as task
ACTIONS=("rest","wander","play")
RECENT_DAYS=1.5
DROP=.22
def gate(c,profile,now,context,prob,goal,*,max_weight=.30):
    original=earlier._unit(prob)
    if goal not in original:
        return {"weights":original,"strength":0.,"reason":"no_goal",
                "effective_goal":goal,"previous_goal":goal,"sources":[]}
    observed=task.evidence(c,profile,now,context,limit=128)
    from_last=[r for r in observed if r["action"]==goal]
    current=task.posterior(observed,list(original),now=now,half_life_days=RECENT_DAYS)
    long=task.posterior(observed,list(original),now=now,half_life_days=9.)
    rec=current["expected_task_success"]
    past=long["expected_task_success"]
    counts=current["counts"]
    source_ids=[r["source"] for r in observed[:12]]
    bad_evidence=False
    if len(from_last)>=16:
        new,old=from_last[:8],from_last[8:16]
        new_r=sum(int(e["passed"]) for e in new)/8
        old_r=sum(int(e["passed"]) for e in old)/8
        bad_evidence=(old_r-new_r>=DROP and
                     (past[goal]-rec[goal]>=.065 or new_r<=.375))
    if len(from_last)>=6 and now-float(from_last[0]["at"])>=2.*86400:
        bad_evidence=True
    goal_gap=rec[goal]-sum(original[a]*rec[a] for a in original)
    if bad_evidence:
        # An *experimental* alternative based strictly on observed outcomes;
        # small-sample alternatives receive bounded optimistic bonuses.
        explore={a:rec[a]+.12/math.sqrt(counts[a]+1.)
                 for a in ACTIONS if a!=goal}
        alternative=max(explore,key=lambda a:(explore[a],-ACTIONS.index(a)))
        # Only switch goal when the alternative has at least one observed
        # outcome and is plausibly better than the degraded incumbent.
        if counts[alternative]>=2 and explore[alternative]>=rec[goal]+.04:
            newgoal=alternative
            strength=.30
            reason="trial_alternative_after_verified_decline"
        else:
            newgoal=goal
            strength=0.
            reason="decline_detected_no_supported_alternative"
    elif counts.get(goal,0)<4 or len(observed)<12:
        strength=0.
        newgoal=goal
        reason="insufficient_graded_experience"
    elif goal_gap>=.08:
        strength=min(max_weight,.19*counts[goal]/(counts[goal]+5.))
        newgoal=goal
        reason="current_goal_confirmed"
    elif goal_gap<-.05:
        strength=0.
        newgoal=goal
        reason="current_goal_not_competitive"
    else:
        strength=.035
        newgoal=goal
        reason="weak_goal_evidence"
    weights=earlier._unit({a:(1-strength)*original[a]+strength*
                           (.82 if a==newgoal else .09) for a in original})
    return {"weights":weights,"strength":round(strength,6),
            "reason":reason,"effective_goal":newgoal,
            "previous_goal":goal,"changed_goal":newgoal!=goal,
            "recent_estimates":{a:round(rec[a],4) for a in ACTIONS},
            "long_estimates":{a:round(past[a],4) for a in ACTIONS},
            "counts":counts,"observations":len(observed),
            "sources":source_ids,"source_ids":source_ids,
            "outcome_oracle_used_for_decision":False}
