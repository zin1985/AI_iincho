"""Lab-only v3 gate: revalidate older 9+3 goal with recent observed evidence."""
from __future__ import annotations
from tools import fourth_goal_adoption_gate_v2 as v2
import brain_9plus3_task_learning as task
AGE_LIMIT_SECONDS=2.5*86400.
DROP=.3

def gate(c,profile,now,context,prob,goal,*,max_weight=.42):
    old=v2.gate(c,profile,now,context,prob,goal,max_weight=max_weight)
    if old["strength"]<=0 or goal not in prob:
        return old
    events=task.evidence(c,profile,now,context,limit=128)
    relevant=[e for e in events if e["action"]==goal]
    reason=None
    if not relevant or now-float(relevant[0]["at"])>AGE_LIMIT_SECONDS:
        reason="goal_evidence_stale"
    elif len(relevant)>=24:
        new,mid,older=relevant[:6],relevant[6:12],relevant[12:24]
        rate=lambda batch:sum(int(e["passed"]) for e in batch)/len(batch)
        if rate(older)-rate(mid)>=DROP and rate(older)-rate(new)>=DROP:
            reason="confirmed_recent_goal_deterioration"
    if reason:
        original=v2._unit(prob)
        old.update({"weights":original,"strength":0.,"reason":reason,
                    "suspended_prior_reason":old["reason"],
                    "goal_evidence_latest_at":(float(relevant[0]["at"]) if relevant else None)})
    return old
