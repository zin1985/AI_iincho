"""Evidence-first validity checks for previously authorized contextual strategies.

Only an expired or objectively contradicted *numeric action bias* is paused.
Never edits memories, goals, authored values or subjective reflections.
Read-only evaluation is safe on every decision. Explicit persona review is
still required to reaffirm/revoke a strategy.
"""
from __future__ import annotations
import math,time

TTL_DAYS=7
MIN_POST=8
MIN_BEFORE=8
DOWNTURN=.22

def assess(c,profile,context_key,at=None):
    import brain_9plus3_task_learning as task
    now=time.time() if at is None else float(at)
    has=c.execute("""select 1 from sqlite_master
                     where type='table' and name='brain_9plus3_persona_strategies'""").fetchone()
    if not has:return None
    row=c.execute("""select s.*,p.created_at as proposal_at
        from brain_9plus3_persona_strategies s
        join brain_9plus3_strategy_proposals p on p.id=s.proposal_id
        where s.profile=? and s.context_key=?""",(profile,context_key)).fetchone()
    if row is None or float(row["updated_at"])>now:return None
    updated=float(row["updated_at"])
    age_days=(now-updated)/86400.
    observations=task.evidence(c,profile,now,context_key,limit=300)
    post=[r for r in observations if r["at"]>updated+0.001]
    pre=[r for r in observations if r["at"]<=updated]
    # Compare observed results for the *authorized action itself*, not
    # another goal/context and not invented counterfactual outcomes.
    action=row["action"]
    before_action=[r for r in pre if r["action"]==action][:16]
    after_action=[r for r in post if r["action"]==action][:16]
    older=before_action[:16]
    recent=after_action[:12]
    earlier_p=(sum(r["passed"] for r in older)/len(older)
               if len(older)>=4 else None)
    newer_p=(sum(r["passed"] for r in recent)/len(recent)
             if len(recent)>=4 else None)
    decline=(len(post)>=MIN_POST and earlier_p is not None and newer_p is not None
             and earlier_p-newer_p>=DOWNTURN)
    # Even if samples for one action are sparse, a substantial context-level
    # downturn is worth asking the review model to reconsider.
    post12=post[:12]
    pre12=pre[:12]
    context_decline=(len(post)>=MIN_POST and len(pre12)>=MIN_BEFORE and
         (sum(x["passed"] for x in pre12)/len(pre12))-
         (sum(x["passed"] for x in post12)/len(post12))>=.3)
    # A single unlucky patch of events is not robust evidence of drift.
    # Require the same signal to have been present six virtual hours ago.
    # The read-only replay uses only data available at the evaluation time.
    earlier_cutoff=now-6*3600.
    earlier_post=[x for x in post if x["at"]<earlier_cutoff]
    earlier_by_action=[x for x in earlier_post if x["action"]==action][:12]
    earlier_chosen_p=(sum(x["passed"] for x in earlier_by_action)/
                    len(earlier_by_action)) if len(earlier_by_action)>=4 else None
    stable_decline=(len(earlier_post)>=MIN_POST and
       earlier_p is not None and earlier_chosen_p is not None and
       earlier_p-earlier_chosen_p>=DOWNTURN)
    earlier_window=earlier_post[:12]
    stable_context_drop=(len(earlier_window)>=MIN_POST and
       len(pre12)>=MIN_BEFORE and
       sum(x["passed"] for x in pre12)/len(pre12)-
       sum(x["passed"] for x in earlier_window)/len(earlier_window)>=.3)
    decline=bool(decline and stable_decline)
    context_decline=bool(context_decline and stable_context_drop)
    expired=age_days>=TTL_DAYS
    status=("expired" if expired else
            "needs_review" if decline or context_decline else "active")
    evidence=(recent[:5] if decline else post12[:5])
    return {"status":status,"context_key":context_key,"action":action,
       "proposal_id":int(row["proposal_id"]),"review_source":row["review_source"],
       "updated_at":updated,"age_days":round(age_days,3),
       "after_count":len(post),"before_action_n":len(before_action),
       "after_action_n":len(after_action),
       "prior_action_success":round(earlier_p,4) if earlier_p is not None else None,
       "recent_action_success":round(newer_p,4) if newer_p is not None else None,
       "decline_detected":bool(decline),"context_drop_detected":bool(context_decline),
       "evidence_sources":[r["source"] for r in evidence],
       "safe_to_apply_bias":status=="active",
       "no_persona_memory_deleted":True}

def monitor(c,profile,at=None,limit=4):
    """Read-only review list; only lists approved-but-now-inactive biases."""
    import brain_9plus3_self_revision as strategies
    now=time.time() if at is None else float(at)
    items=strategies.current(c,profile,at=now,limit=limit)
    result=[]
    for item in items:
        state=assess(c,profile,item["context_key"],now)
        if state and state["status"]!="active":result.append(state)
    return result

def ensure(c):
    c.execute("""create table if not exists brain_9plus3_strategy_recheck_decisions(
      id integer primary key,profile text not null,context_key text not null,
      proposal_id integer not null,review_id integer,
      decision text not null,reason text not null,source_kind text not null,
      evidence_sources_json text not null,at real not null,
      unique(profile,context_key,review_id))""")

def decide(c,profile,review_id,context_key,decision,*,at=None,
           reason="",synthetic_test_only=False):
    """Reviewer can explicitly revoke or reconsider a stale numeric bias.

    Reaffirmed strategy has a refreshed start time; if evidence still shows
    a decline on the next round, guard suppresses its bias again.
    """
    import brain_9plus3_self_revision as revision
    import json
    if decision not in ("revoke","reaffirm","defer"):return {"status":"invalid_decision"}
    now=time.time() if at is None else float(at)
    ensure(c)
    status=assess(c,profile,context_key,at=now)
    if not status:return {"status":"missing_strategy"}
    if status["status"]=="active":return {"status":"not_due"}
    if synthetic_test_only:
        if profile!="sim_fourth":return {"status":"test_only_sim_fourth"}
        source_kind="synthetic_review_not_N"
    else:
        if profile!="iincho_n" or review_id is None:return {"status":"unauthorized"}
        proof=c.execute("""select id from self_growth_reviews where id=?
          and profile='iincho_n' and status='reserved' and ts>=?""",
          (review_id,status["updated_at"])).fetchone()
        if not proof:return {"status":"not_authorized_review"}
        source_kind="persona_model_review"
    prev=c.execute("""select 1 from brain_9plus3_strategy_recheck_decisions
       where profile=? and context_key=? and review_id is ?""",
       (profile,context_key,review_id)).fetchone()
    if prev:return {"status":"already_decided"}
    c.execute("""insert into brain_9plus3_strategy_recheck_decisions(
        profile,context_key,proposal_id,review_id,decision,reason,
        source_kind,evidence_sources_json,at) values(?,?,?,?,?,?,?,?,?)""",
        (profile,context_key,status["proposal_id"],review_id,decision,
         str(reason)[:300],source_kind,json.dumps(status["evidence_sources"]),now))
    if decision=="revoke":
        c.execute("""delete from brain_9plus3_persona_strategies
                  where profile=? and context_key=? and proposal_id=?""",
                  (profile,context_key,status["proposal_id"]))
        # Records and authentic persona memories are retained.
    elif decision=="reaffirm":
        c.execute("""update brain_9plus3_persona_strategies
             set updated_at=?,review_source=? where profile=? and context_key=?
                and proposal_id=?""",(now,source_kind+"#"+str(review_id or "sandbox"),
                                       profile,context_key,status["proposal_id"]))
    return {"status":decision,"context_key":context_key,
            "recorded_sources":status["evidence_sources"],
            "numeric_bias_revoked":decision=="revoke","memories_deleted":False}