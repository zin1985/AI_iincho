"""Objective task outcome learning for the nine+three / tiny-brain loop.

A valid virtual action is not necessarily a successful task.
Only verified evaluator results, linked to a real world_event, train this
contextual bandit. This module does NOT assign subjective feelings or
authored preferences, and never fabricates rewards for unchosen actions.
Production N requires an explicitly verified non-simulated evaluator.
"""
from __future__ import annotations
import json, math

ACTIONS=("rest","wander","play","travel_to_project","advance_project")
TABLE="brain_9plus3_task_outcomes"
VERSION="objective_result_contextual_bandit_v1"

def ensure(c):
    c.execute("""create table if not exists brain_9plus3_task_outcomes(
      id integer primary key,profile text not null,ts real not null,
      context_key text not null,action text not null,
      event_id integer not null unique,task_success integer not null,
      outcome_score real not null, predicted_prior real,
      source_kind text not null, evaluator_id text not null)""")
    c.execute("create index if not exists ix_9p3_objective_context on brain_9plus3_task_outcomes(profile,context_key,ts)")

def record(c,profile,at,context_key,action,event_id,success,
           *,source_kind,evaluator_id,predicted_prior=None):
    """Caller supplies externally evaluated task success. Never infer it."""
    if action not in ACTIONS:raise ValueError("unknown action")
    if not isinstance(context_key,str) or not 4<=len(context_key)<=120:
        raise ValueError("task context required")
    if source_kind not in ("simulation_environment","verified_task_evaluator"):
        raise ValueError("untrusted task outcome source")
    if source_kind=="simulation_environment" and profile!="sim_fourth":
        raise ValueError("synthetic outcome cannot train a real persona")
    if not isinstance(success,bool):raise ValueError("task outcome must be boolean")
    if not evaluator_id:raise ValueError("missing evaluator identity")
    row=c.execute("""select id,profile,ts from world_events
                     where id=? and profile=?""",(event_id,profile)).fetchone()
    if row is None or abs(float(row["ts"])-float(at))>1.:
        raise ValueError("task outcome must link to same person's actual action")
    ensure(c)
    prev=c.execute("""select id,profile,context_key,action,task_success,
                    source_kind,evaluator_id from brain_9plus3_task_outcomes
                      where event_id=?""",(event_id,)).fetchone()
    if prev:
        same=(prev["profile"]==profile and prev["context_key"]==context_key and
              prev["action"]==action and prev["task_success"]==int(success) and
              prev["source_kind"]==source_kind and prev["evaluator_id"]==evaluator_id)
        if not same:raise ValueError("conflicting result for already evaluated world action")
        return {"id":int(prev["id"]),"duplicate":True}
    inserted=c.execute("""insert into brain_9plus3_task_outcomes(
      profile,ts,context_key,action,event_id,task_success,outcome_score,
      predicted_prior,source_kind,evaluator_id)
      values(?,?,?,?,?,?,?,?,?,?)""",
      (profile,float(at),context_key,action,int(event_id),int(success),
       float(success),predicted_prior,source_kind,evaluator_id))
    return {"id":int(inserted.lastrowid),"duplicate":False}

def evidence(c,profile,at,context_key,*,limit=128):
    if c.execute("select 1 from sqlite_master where type='table' and name=?",
                 (TABLE,)).fetchone() is None:return []
    rows=c.execute("""select o.id,o.ts,o.action,o.task_success,o.event_id,
               o.source_kind,o.predicted_prior from brain_9plus3_task_outcomes o
        join world_events e on e.id=o.event_id and e.profile=o.profile
        where o.profile=? and o.context_key=? and o.ts<?
        and o.ts>=? and (?='sim_fourth' or o.source_kind='verified_task_evaluator')
        order by o.ts desc,o.id desc limit ?""",
        (profile,context_key,float(at),float(at)-15*86400,profile,
         max(1,min(300,int(limit))))).fetchall()
    return [{"id":int(r["id"]),"source":TABLE+"#"+str(r["id"]),
             "action":r["action"],"passed":bool(r["task_success"]),
             "at":float(r["ts"]),"event_id":r["event_id"],
             "kind":r["source_kind"],
             "prior_success_prediction":r["predicted_prior"]}
            for r in rows]

def posterior(events,actions,*,now=None,half_life_days=0.):
    """Beta-Bernoulli or time-decayed Beta for changed task conditions."""
    counts={a:0 for a in actions};wins={a:0 for a in actions}
    effective={a:0. for a in actions};weighted_wins={a:0. for a in actions}
    if half_life_days<0:raise ValueError("half life cannot be negative")
    for r in events:
        a=r["action"]
        if a not in counts:continue
        counts[a]+=1
        wins[a]+=int(r["passed"])
        weight=(2.**(-max(0.,float(now)-float(r["at"]))/
                       (float(half_life_days)*86400.))
                if now is not None and half_life_days else 1.)
        effective[a]+=weight
        weighted_wins[a]+=weight*int(r["passed"])
    estimate={a:(weighted_wins[a]+1.)/(effective[a]+2.) for a in actions}
    return {"counts":counts,"successes":wins,"expected_task_success":estimate,
            "effective_observations":round(sum(effective.values()),5),
            "total_observed":sum(counts.values())}

def adapt(c,profile,at,context_key,base,*,active=True,half_life_days=0.):
    original={a:float(v) for a,v in base.items()}
    observations=evidence(c,profile,at,context_key) if active else []
    estimate=posterior(observations,list(original),now=at,
                       half_life_days=half_life_days)
    n=estimate["effective_observations"]
    if not observations:
        return {"baseline":original,"updated":original,"sources":[],
                "observed":0,"blend":0.,"posterior":estimate,
                "prediction_errors":[],"max_shift":0.,
                "no_unobserved_rewards":True,"version":VERSION,
            "half_life_days":half_life_days}
    # The outcome head can eventually override a strong imitation prior,
    # but only when the current task has sufficient *verified* experiences.
    strength=min(.86,.92*n/(n+9.))
    vals=estimate["expected_task_success"]
    # Boltzmann distribution of learned rewards, not action hardcoding.
    raw={a:math.exp(5.*(v-.5)) for a,v in vals.items()}
    total=sum(raw.values())
    objective={a:v/total for a,v in raw.items()}
    updated={a:(1-strength)*original[a]+strength*objective[a]
             for a in original}
    errors=[]
    for e in observations[:8]:
        if e["prior_success_prediction"] is not None:
            errors.append(round(float(e["passed"])-float(e["prior_success_prediction"]),4))
    return {"baseline":original,"updated":updated,
            "sources":[x["source"] for x in observations],
            "observed":n,"blend":round(strength,5),
            "posterior":estimate,"prediction_errors":errors,
            "max_shift":round(max(abs(updated[a]-original[a]) for a in original),6),
            "no_unobserved_rewards":True,"version":VERSION,
            "half_life_days":half_life_days}
