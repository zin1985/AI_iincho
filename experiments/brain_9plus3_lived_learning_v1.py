"""Persistent evidence-backed nine-region / three-integrator learning.

A computed region's state is NOT an emotion or biological brain state.
Each row is an audit of a REAL verified virtual action or a graded simulated
task outcome. Real N cannot receive simulation_environment records.

The nine processors have independent time-persistent values, explicit
cross-region dependencies, and three downstream integrators. Only objective
observations can be inputs. No hidden oracle or unchosen reward is read.
"""
from __future__ import annotations
import json,math
REGIONS=("kaiba","hentou","zentou","taijou","shounou","shichou",
         "shichoukashibu","toutai","naijima")
INTEGRATORS=("reward_value","world_causal_model","meta_reflection")
ACTIONS=("rest","wander","play","travel_to_project","advance_project")
LINKS={"zentou":("kaiba","shichou","shichoukashibu"),
       "shichou":("hentou","naijima"),
       "toutai":("taijou","shounou"),
       "reward_value":("hentou","zentou","shichoukashibu"),
       "world_causal_model":("kaiba","shounou","shichou"),
       "meta_reflection":("toutai","taijou","naijima","zentou")}
ALPHA=.24

def ensure(c):
    c.execute("""create table if not exists brain_9plus3_lived_events(
      id integer primary key,profile text not null,context_key text not null,
      source_id text not null unique,source_kind text not null,
      action text not null,operational_applied integer,
      task_success integer,ts real not null,region_json text not null,
      integrators_json text not null,links_json text not null)""")
    c.execute("""create table if not exists brain_9plus3_region_states(
      profile text not null,context_key text not null,region text not null,
      value real not null,events integer not null,updated_at real not null,
      last_source_id text not null,
      primary key(profile,context_key,region))""")
    c.execute("""create table if not exists brain_9plus3_integrator_states(
      profile text not null,context_key text not null,integrator text not null,
      value real not null,events integer not null,updated_at real not null,
      last_source_id text not null,
      primary key(profile,context_key,integrator))""")
    c.execute("""create table if not exists brain_9plus3_goal_choices(
      id integer primary key,profile text not null,context_key text not null,
      chosen_goal text not null,previous_goal text,
      reason_json text not null,source_ids_json text not null,at real not null,
      synthetic_model_review integer not null default 0)""")
    c.execute("create index if not exists idx_nine3_lived_scope on brain_9plus3_lived_events(profile,context_key,ts)")

def _clamp(v):
    return max(0.,min(1.,float(v)))

def _state(c,table,profile,context,keys):
    rows=c.execute("select * from "+table+" where profile=? and context_key=?",
                   (profile,context)).fetchall()
    data={r["region"] if table.endswith("region_states") else r["integrator"]:r
          for r in rows}
    return {k:(float(data[k]["value"]) if k in data else None) for k in keys}

def _blend(c,table,profile,context,key,value,now,source):
    column="region" if table.endswith("region_states") else "integrator"
    old=c.execute("select value,events from "+table+
        " where profile=? and context_key=? and "+column+"=?",
        (profile,context,key)).fetchone()
    val=_clamp(value)
    if old:
        val=_clamp((1-ALPHA)*float(old["value"])+ALPHA*val)
    c.execute("insert into "+table+
        " (profile,context_key,"+column+",value,events,updated_at,last_source_id)"
        " values(?,?,?,?,?,?,?) on conflict(profile,context_key,"+column+")"
        " do update set value=excluded.value,events=excluded.events,"
        " updated_at=excluded.updated_at,last_source_id=excluded.last_source_id",
        (profile,context,key,val,(int(old["events"])+1 if old else 1),now,source))
    return round(val,5)

def _prior(c,profile,context,at,limit=36):
    return c.execute("""select action,task_success,operational_applied,ts,source_id
       from brain_9plus3_lived_events where profile=? and context_key=?
       and ts<? order by ts desc,id desc limit ?""",
       (profile,context,float(at),limit)).fetchall()

def observe_task(c,profile,context,action,at,source_id,success,prior_prediction=None,
                 *,source_kind="verified_task_evaluator"):
    if source_kind=="simulation_environment" and profile!="sim_fourth":
        raise ValueError("synthetic grade must not enter production persona")
    if source_kind not in ("verified_task_evaluator","simulation_environment"):
        raise ValueError("unsupported result provenance")
    if type(success) is not bool or action not in ACTIONS:
        raise ValueError("verified task success and known action required")
    return _observe(c,profile,context,action,at,source_id,
                    task_success=int(success),operational_applied=None,
                    source_kind=source_kind,prediction=prior_prediction)

def observe_operational(c,profile,context,action,at,source_id,applied):
    if type(applied) is not bool or action not in ACTIONS:
        raise ValueError("operational observation requires action and result")
    # An executed action is NOT proof of task success. No graded reward.
    return _observe(c,profile,context,action,at,source_id,
                    task_success=None,operational_applied=int(applied),
                    source_kind="verified_operational_action",prediction=None)

def _observe(c,profile,context,action,at,source_id,task_success,
             operational_applied,source_kind,prediction):
    ensure(c)
    if not isinstance(context,str) or not 4<=len(context)<=120:
        raise ValueError("context is required")
    old=c.execute("select id,profile,context_key,action,source_kind from brain_9plus3_lived_events where source_id=?",
                  (source_id,)).fetchone()
    if old:
        if (old["profile"],old["context_key"],old["action"],old["source_kind"])!=(
             profile,context,action,source_kind):
            raise ValueError("conflicting source ID")
        return {"id":int(old["id"]),"duplicate":True}
    previous=_prior(c,profile,context,at)
    graded=[x for x in previous if x["task_success"] is not None]
    action_history=[x for x in graded if x["action"]==action]
    counts={a:sum(x["action"]==a for x in graded) for a in ACTIONS}
    wins={a:sum(int(x["task_success"]) for x in graded if x["action"]==a)
          for a in ACTIONS}
    observed=sum(counts.values())
    posterior={a:(wins[a]+1.)/(counts[a]+2.) for a in ACTIONS}
    # Distinguish task utility from operational validity.
    outcome=task_success if task_success is not None else operational_applied
    freq=sum(x["action"]==action for x in previous)/max(1,len(previous))
    prediction=(float(prediction) if prediction is not None and
                0<=float(prediction)<=1 else None)
    error=abs(outcome-prediction) if prediction is not None and task_success is not None else None
    habit=max(ACTIONS,key=lambda a:sum(x["action"]==a for x in previous)) if previous else None
    best=max(ACTIONS,key=lambda a:posterior[a]) if observed>=6 else None
    conflict=float(best is not None and habit is not None and best!=habit)
    recent_fail=sum(x["task_success"]==0 for x in graded[:12])
    previous_states=_state(c,"brain_9plus3_region_states",profile,context,REGIONS)
    known={k:(previous_states[k] if previous_states[k] is not None else 0.5)
           for k in REGIONS}
    # Nine semantically separate measurable features.
    raw={
      "kaiba":min(1.,len(previous)/24.),
      "hentou":1. if outcome==0 else min(1.,recent_fail/12.),
      "zentou":posterior[action] if task_success is not None else .5,
      "taijou":freq,
      "shounou":error if error is not None else 0.,
      "shichou":1./math.sqrt(1+counts[action]),
      "shichoukashibu":min(1.,sum(x["ts"]>=at-86400 for x in previous)/24.),
      "toutai":conflict,
      "naijima":1./math.sqrt(1+observed)}
    # Dependent processors read *prior* values, not a circular same-cycle value.
    raw["zentou"]=_clamp(.64*raw["zentou"]+.15*known["kaiba"]+
                            .11*known["shichou"]+.10*(1-known["shichoukashibu"]))
    raw["shichou"]=_clamp(.7*raw["shichou"]+.16*known["hentou"]+
                           .14*known["naijima"])
    raw["toutai"]=_clamp(.58*raw["toutai"]+.24*known["shounou"]+
                          .18*known["taijou"])
    regions={k:_blend(c,"brain_9plus3_region_states",profile,context,
                      k,raw[k],at,source_id) for k in REGIONS}
    integrator_raw={
      "reward_value":(.42*(outcome if task_success is not None else .5)
                      +.33*regions["zentou"]+.15*(1-regions["hentou"])
                      +.10*(1-regions["shichoukashibu"])),
      "world_causal_model":(.42*(1-regions["shounou"])
                            +.3*regions["kaiba"]+.28*regions["shichou"]),
      "meta_reflection":(.42*regions["toutai"]+.28*regions["taijou"]
                         +.18*regions["naijima"]+.12*(1-regions["zentou"]))}
    integrators={k:_blend(c,"brain_9plus3_integrator_states",profile,
                            context,k,v,at,source_id)
                 for k,v in integrator_raw.items()}
    # Audit explicit connections between role-specific computations.
    edges=[{"from":src,"to":dst,"prior_value":round(known[src],5)}
           for dst,sources in LINKS.items() for src in sources]
    out=c.execute("""insert into brain_9plus3_lived_events(
      profile,context_key,source_id,source_kind,action,operational_applied,
      task_success,ts,region_json,integrators_json,links_json)
      values(?,?,?,?,?,?,?,?,?,?,?)""",
      (profile,context,source_id,source_kind,action,operational_applied,
       task_success,at,json.dumps(regions),json.dumps(integrators),
       json.dumps(edges)))
    return {"id":int(out.lastrowid),"duplicate":False,"region_states":regions,
            "integrators":integrators,"links":edges,
            "source_id":source_id,"task_grade_known":task_success is not None,
            "not_biological_or_subjective":True}

def snapshot(c,profile,context):
    if not c.execute("select 1 from sqlite_master where type='table' and name='brain_9plus3_region_states'").fetchone():
        return {"regions":{},"integrators":{},"sources":[]}
    regions=_state(c,"brain_9plus3_region_states",profile,context,REGIONS)
    integrators=_state(c,"brain_9plus3_integrator_states",profile,context,INTEGRATORS)
    sources=[r["source_id"] for r in c.execute("""select source_id
      from brain_9plus3_lived_events where profile=? and context_key=?
      order by ts desc,id desc limit 12""",(profile,context))]
    return {"regions":regions,"integrators":integrators,"sources":sources,
            "is_subjective_experience":False,
            "all_nine_defined":all(v is not None for v in regions.values())}
