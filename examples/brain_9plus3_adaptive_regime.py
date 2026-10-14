"""Sandbox-only, evidenced regime-change adaptation for the fourth persona.

Detects persistent outcome deterioration over disjoint chronological windows
in a single task. It never uses the hidden oracle, unchosen counterfactuals,
subjective moods, or model-generated fabricated achievements.

Return a proposal for exploration and recency: the caller does not alter
authored goals/values, only ephemeral numeric decision weights.
"""
from __future__ import annotations
from brain_9plus3_task_learning import evidence

BASE_EPSILON=.22
ACTIVE_EPSILON=.32
RECENT_HALF_LIFE_DAYS=1.5
MIN_BASELINE=12
WINDOW=6
DROP=.40

def advise(c,profile,now,context,*,base_epsilon=BASE_EPSILON):
    events=evidence(c,profile,now,context,limit=128)
    if len(events)<MIN_BASELINE+2*WINDOW:
        return {"changed":False,"epsilon":base_epsilon,"half_life_days":0.,
                "reason":"insufficient_experience","sources":[]}
    # "events" are most recent first. Crucially all three windows use only
    # action/outcome pairs that were actually chosen before now.
    newest=events[:WINDOW]
    previous=events[WINDOW:2*WINDOW]
    historical=events[2*WINDOW:2*WINDOW+24]
    if len(historical)<MIN_BASELINE:
        return {"changed":False,"epsilon":base_epsilon,"half_life_days":0.,
                "reason":"insufficient_baseline","sources":[]}
    def mean(xs):
        return sum(int(e["passed"]) for e in xs)/len(xs)
    baseline=mean(historical)
    recent=mean(newest)
    prior=mean(previous)
    # A change in action mix can mimic regime changes. Require a viable
    # baseline sample on an action represented in *each* recent window.
    shared={e["action"] for e in newest}&{e["action"] for e in previous}
    same_action=[]
    for a in shared:
        old=[e for e in historical if e["action"]==a]
        mid=[e for e in previous if e["action"]==a]
        new=[e for e in newest if e["action"]==a]
        if len(old)>=8 and len(mid)>=4 and len(new)>=4:
            same_action.append((a,mean(old),mean(mid),mean(new),old,mid,new))
    convincer=[t for t in same_action
               if t[1]>=.65 and t[1]-t[2]>=DROP and t[1]-t[3]>=DROP
               and t[2]<=.35 and t[3]<=.35]
    if convincer:
        hit=max(convincer,key=lambda x:(x[1]-x[2])+(x[1]-x[3]))
        return {"changed":True,"epsilon":ACTIVE_EPSILON,
                "half_life_days":RECENT_HALF_LIFE_DAYS,
                "reason":"same_action_persistent_regime_deterioration",
                "context":context,"action":hit[0],
                "old_rate":round(hit[1],3),"middle_rate":round(hit[2],3),
                "latest_rate":round(hit[3],3),
                "sources":[e["source"] for e in hit[4]+hit[5]+hit[6]],
                "subjective_feelings_inferred":False}
    return {"changed":False,"epsilon":base_epsilon,"half_life_days":0.,
            "reason":"no_persistent_same_action_deterioration",
            "context_rate_window":[round(baseline,3),round(prior,3),round(recent,3)],
            "sources":[]}

def rebalance(c,profile,now,context,prior,*,declining_action=None):
    """Change actual action ranking after corroborated drift, not just epsilon."""
    import math
    from brain_9plus3_task_learning import evidence,posterior
    original=dict(prior)
    observations=evidence(c,profile,now,context,limit=128)
    recent=posterior(observations,list(original),now=now,
                     half_life_days=RECENT_HALF_LIFE_DAYS)
    estimates=recent["expected_task_success"]
    raw={a:math.exp(5.*(p-.5)) for a,p in estimates.items()}
    sum_raw=sum(raw.values())
    if sum_raw<=0:return original
    # Bound the degree of adjustment, preserve every action's nonzero chance.
    adaptation=.55
    updated={a:(1-adaptation)*original[a]+
                adaptation*raw[a]/sum_raw for a in original}
    # The observed failing incumbent should not keep monopolizing the
    # next experiments after two independently confirmed bad windows.
    # The other alternatives remain symmetric unless observed rewards
    # distinguish them; no hidden right answer is supplied.
    if declining_action in updated and len(updated)>1:
        updated[declining_action]*=.28
        denom=sum(updated.values())
        updated={a:v/denom for a,v in updated.items()}
    return updated