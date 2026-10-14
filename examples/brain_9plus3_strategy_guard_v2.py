"""Stricter P0 experimental strategy guard, with two independent evidence windows.

This module is intentionally NOT production-wired. It observes only the
verified outcomes of a formerly N-approved action in the same task.
No oracle, unselected actions, private thoughts or persona data changes.
"""
from __future__ import annotations
import time
from brain_9plus3_task_learning import evidence
from brain_9plus3_self_revision import current

WINDOW=6
MIN_BASELINE=8
DROP=.30
FRESHNESS_DAYS=7
MAX_AGE_DAYS=21

def assess(c,profile,context,at):
    now=float(at)
    approved=current(c,profile,context,at=now,limit=1)
    if not approved:return None
    entry=approved[0]
    approved_at=float(entry["updated_at"])
    action=entry["action"]
    data=evidence(c,profile,now,context,limit=300)
    same=[e for e in data if e["action"]==action]
    before=[e for e in same if e["at"]<=approved_at][:24]
    newer=[e for e in same if e["at"]>approved_at]
    baseline=(sum(x["passed"] for x in before[:24])/len(before[:24])
              if len(before)>=MIN_BASELINE else None)
    window_a=newer[:WINDOW]
    window_b=newer[WINDOW:2*WINDOW]
    def rate(rows):
        return sum(int(x["passed"]) for x in rows)/len(rows) if len(rows)>=WINDOW else None
    a,b=rate(window_a),rate(window_b)
    both=(baseline is not None and a is not None and b is not None
          and baseline-a>=DROP and baseline-b>=DROP)
    latest_at=same[0]["at"] if same else approved_at
    stale_evidence=now-latest_at>=FRESHNESS_DAYS*86400
    hard_timeout=now-approved_at>=MAX_AGE_DAYS*86400
    verdict=("contradicted" if both else
             "stale_evidence" if stale_evidence else
             "maximum_age" if hard_timeout else "active")
    return {"status":verdict,"can_apply_bias":verdict=="active",
        "profile":profile,"context":context,"action":action,
        "proposal_id":entry["proposal_id"],"review_source":entry["review_source"],
        "baseline_n":len(before),"new_same_action_n":len(newer),
        "baseline_success":round(baseline,4) if baseline is not None else None,
        "recent_window_success":round(a,4) if a is not None else None,
        "earlier_window_success":round(b,4) if b is not None else None,
        "stale_days":round((now-latest_at)/86400,3),
        "evidence_sources":[e["source"] for e in newer[:2*WINDOW]],
        "no_memory_mutation":True}

def guarded_bias(original_bias,c,profile,context,at,prob):
    """Sandbox adapter for instrumented A/B. Production policy unchanged."""
    standard=original_bias(c,profile,context,at,prob)
    safety=assess(c,profile,context,at)
    if safety and not safety["can_apply_bias"]:
        return {"weights":dict(prob),"approved_sources":[],
                "changed":False,"suppression":safety}
    return standard