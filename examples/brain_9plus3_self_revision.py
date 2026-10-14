"""9+3 -> explicit self-revision proposal -> authentic review -> future policy.

A task success is not a personal feeling or value. Nothing changes the
durable self-model automatically. The existing persona daily-review model
may accept, reject or defer a strategy candidate grounded in verified
task outcomes. Synthetic sandbox experiments are separately flagged.
"""
from __future__ import annotations
import json
import sqlite3
import time

ACTIONS=("rest","wander","play","travel_to_project","advance_project")
DECISIONS=("adopt","defer","reject")
VERSION="nine_three_self_revision_1"

def _has(c,table):
    return c.execute("select 1 from sqlite_master where type='table' and name=?",(table,)).fetchone() is not None

def ensure(c):
    c.execute("""create table if not exists brain_9plus3_strategy_proposals(
      id integer primary key,profile text not null,context_key text not null,
      suggested_action text not null,reason text not null,
      evidence_json text not null,source_count integer not null,
      created_at real not null,expires_at real not null,
      version text not null,unique(profile,context_key,evidence_json))""")
    c.execute("""create table if not exists brain_9plus3_strategy_decisions(
      id integer primary key,proposal_id integer not null,
      profile text not null,review_id integer,
      decision text not null,reason text not null,
      source_kind text not null,at real not null,
      unique(proposal_id,profile,review_id))""")
    c.execute("""create table if not exists brain_9plus3_persona_strategies(
      profile text not null,context_key text not null,
      action text not null,confidence real not null,
      review_source text not null,proposal_id integer not null,
      updated_at real not null,
      primary key(profile,context_key))""")
    c.execute("create index if not exists ix_9plus3_persona_proposals on brain_9plus3_strategy_proposals(profile,created_at desc)")

def propose(c,profile,context_key,*,at=None):
    """Draft only when independent task observations justify a comparison."""
    import brain_9plus3_task_learning as task
    if not isinstance(context_key,str) or not 4<=len(context_key)<=120:
        return None
    now=time.time() if at is None else float(at)
    events=task.evidence(c,profile,now,context_key,limit=128)
    # Never consume experimental data for N or other real personalities.
    if profile!="sim_fourth" and any(x["kind"]!="verified_task_evaluator" for x in events):
        return None
    if len(events)<12:return None
    observed={}
    for event in events:
        bucket=observed.setdefault(event["action"],[0,0])
        bucket[0]+=int(event["passed"]);bucket[1]+=1
    eligible={a:((wins+1)/(count+2),count) for a,(wins,count) in observed.items()
              if count>=3 and a in ACTIONS}
    if len(eligible)<2:return None
    order=sorted(eligible,key=lambda a:(-eligible[a][0],a))
    best,second=order[:2]
    difference=eligible[best][0]-eligible[second][0]
    if difference<.12:return None
    sources=[x["source"] for x in events[:32]]
    fingerprint=json.dumps(sources,ensure_ascii=False,separators=(",",":"))
    ensure(c)
    stored=c.execute("""select id from brain_9plus3_strategy_proposals
        where profile=? and context_key=? and evidence_json=?""",
        (profile,context_key,fingerprint)).fetchone()
    if stored:return int(stored["id"])
    # Limit candidate churn: once/day/context, and only source-backed.
    recent=c.execute("""select id from brain_9plus3_strategy_proposals
        where profile=? and context_key=? and created_at>?
        order by id desc limit 1""",(profile,context_key,now-86400)).fetchone()
    if recent:return int(recent["id"])
    summary=("課題 "+context_key+" では観測済みの行動 "+best+" の成功推定"+
             str(round(eligible[best][0]*100,1))+"%が比較対象"+
             second+" の"+str(round(eligible[second][0]*100,1))+"%を上回る。"+
             "これは課題依存の仮説であり、本人の好き嫌いではない。")
    row=c.execute("""insert into brain_9plus3_strategy_proposals(
       profile,context_key,suggested_action,reason,evidence_json,
       source_count,created_at,expires_at,version)
       values(?,?,?,?,?,?,?,?,?)""",
       (profile,context_key,best,summary,fingerprint,len(events),
        now,now+14*86400,VERSION))
    return int(row.lastrowid)

def pending(c,profile,*,at=None,limit=3):
    if not _has(c,"brain_9plus3_strategy_proposals"):return []
    now=time.time() if at is None else float(at)
    return [dict(r) | {"source":"brain_9plus3_strategy_proposals#"+str(r["id"]),
                       "approved":False,"user_experience":False}
            for r in c.execute("""select p.id,p.context_key,p.suggested_action,
              p.reason,p.source_count,p.created_at,p.evidence_json
              from brain_9plus3_strategy_proposals p
              where p.profile=? and p.created_at<=? and p.expires_at>=?
                and not exists(select 1 from brain_9plus3_strategy_decisions d
                               where d.proposal_id=p.id and d.profile=p.profile)
              order by p.id desc limit ?""",
              (profile,now,now,max(1,min(5,int(limit)))))]

def decide(c,profile,review_id,decision,*,proposal_id,reason="",at=None,
           synthetic_test_only=False):
    """Apply only an explicit review choice, atomically in review transaction."""
    now=time.time() if at is None else float(at)
    ensure(c)
    if decision not in DECISIONS:return {"status":"invalid_decision"}
    try:pid=int(proposal_id)
    except (ValueError,TypeError):return {"status":"invalid_proposal"}
    candidate=c.execute("""select * from brain_9plus3_strategy_proposals
       where id=? and profile=? and created_at<=? and expires_at>=?""",
       (pid,profile,now,now)).fetchone()
    if not candidate:return {"status":"unavailable"}
    # Review itself must be an actual reserved N-model review, not
    # an unauthenticated API action or a fabricated algorithmic note.
    if not synthetic_test_only:
        if profile!="iincho_n" or review_id is None:return {"status":"unauthorized"}
        review=c.execute("""select id,profile,status,ts from self_growth_reviews
          where id=? and profile=? and status='reserved' and ts>=?""",
          (int(review_id),profile,candidate["created_at"])).fetchone()
        if not review:return {"status":"unverified_review"}
        source_kind="persona_model_daily_review"
    else:
        if profile!="sim_fourth":return {"status":"test_profile_required"}
        source_kind="synthetic_sandbox_review"
    old=c.execute("""select id from brain_9plus3_strategy_decisions
       where proposal_id=? and profile=?""",(pid,profile)).fetchone()
    if old:return {"status":"already_decided"}
    inserted=c.execute("""insert into brain_9plus3_strategy_decisions(
        proposal_id,profile,review_id,decision,reason,source_kind,at)
        values(?,?,?,?,?,?,?)""",
        (pid,profile,review_id,decision,str(reason)[:320],source_kind,now))
    if decision=="adopt":
        # 0.06..0.18 logit-sized weight. Explicitly scoped to a task.
        confidence=min(.18,.06+.002*min(60,int(candidate["source_count"])))
        c.execute("""insert into brain_9plus3_persona_strategies(
          profile,context_key,action,confidence,review_source,
          proposal_id,updated_at) values(?,?,?,?,?,?,?)
          on conflict(profile,context_key) do update set
          action=excluded.action,confidence=excluded.confidence,
          review_source=excluded.review_source,
          proposal_id=excluded.proposal_id,updated_at=excluded.updated_at""",
          (profile,candidate["context_key"],candidate["suggested_action"],
           confidence,source_kind+"#"+str(review_id if review_id else inserted.lastrowid),
           pid,now))
    return {"status":decision,"proposal_id":pid,
            "self_model_changed":decision=="adopt",
            "review_source_kind":source_kind}

def current(c,profile,context_key=None,*,at=None,limit=4):
    if not _has(c,"brain_9plus3_persona_strategies"):return []
    now=time.time() if at is None else float(at)
    params=[profile,now]
    extra=""
    if context_key is not None:
        extra=" and s.context_key=?"
        params.append(context_key)
    params.append(max(1,min(10,int(limit))))
    result=[]
    for r in c.execute("""select s.context_key,s.action,s.confidence,
          s.review_source,s.proposal_id,s.updated_at
          from brain_9plus3_persona_strategies s
          where s.profile=? and s.updated_at<=?""" + extra+
          """ order by s.updated_at desc limit ?""",params):
        result.append(dict(r))
    return result

def bias(c,profile,context_key,at,action_probabilities):
    """Read-only, bounded, N-approved strategy affects same-task selection."""
    initial=dict(action_probabilities)
    if not context_key:return {"weights":initial,"approved_sources":[],"changed":False}
    approved=current(c,profile,context_key,at=at,limit=1)
    if not approved:return {"weights":initial,"approved_sources":[],"changed":False}
    strategy=approved[0]
    target=strategy["action"]
    if target not in initial:return {"weights":initial,"approved_sources":[],"changed":False}
    # Increase logit, not probability directly; keep all actions eligible.
    import math
    raw={a:float(p)*math.exp(strategy["confidence"] if a==target else 0.)
         for a,p in initial.items()}
    total=sum(raw.values())
    if total<=0:return {"weights":initial,"approved_sources":[],"changed":False}
    return {"weights":{a:v/total for a,v in raw.items()},
            "approved_sources":[strategy["review_source"],
                                "brain_9plus3_strategy_proposals#"+str(strategy["proposal_id"])],
            "changed":True,"approved_action":target}

def prepare(db_path,*,at=None,profile="iincho_n"):
    """Generate N-only actionable candidates before the existing daily review.

    No synthetic grade, no extra LLM, no spontaneous self-model mutation.
    """
    if profile!="iincho_n":return {"candidates":0,"reason":"N_only"}
    import world_space
    now=time.time() if at is None else float(at)
    with world_space._connection(db_path) as c:
        c.execute("begin immediate")
        ensure(c)
        task_table="brain_9plus3_task_outcomes"
        contexts=[]
        if _has(c,task_table):
            contexts=[r["context_key"] for r in c.execute("""select context_key
                from brain_9plus3_task_outcomes where profile=?
                and source_kind='verified_task_evaluator' and ts<=? and ts>=?
                group by context_key order by max(ts) desc limit 3""",
                (profile,now,now-14*86400))]
        for ctx in contexts:
            propose(c,profile,ctx,at=now)
        return {"candidates":len(pending(c,profile,at=now)),
                "approved_strategies":len(current(c,profile,at=now)),
                "new_api_calls":0}

def context_for_ai(db_path,*,at=None,limit=3):
    """Preview N-approved strategies and offered proposals only to N."""
    import world_space
    now=time.time() if at is None else float(at)
    with world_space._connection(db_path) as c:
        drafts=pending(c,"iincho_n",at=now,limit=limit)
        approved=current(c,"iincho_n",at=now,limit=limit)
    return {
      "pending_machine_proposals":[
         {"id":d["id"],"source":d["source"],"context":d["context_key"],
          "action":d["suggested_action"],"evidence_count":d["source_count"],
          "reason":d["reason"],
          "verified_sources":json.loads(d["evidence_json"])[:12],
          "not_N_written":True}
         for d in drafts],
      "persona_approved_contextual_strategies":approved,
      "interpretation":"採点済みの課題成績に基づく方針案。採用・保留・拒否はN本人の既存の日次AI判断が決める。価値観や感情は自動で書き換えない。",
      "new_llm_requests":0
    }