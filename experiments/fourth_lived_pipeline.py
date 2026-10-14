"""Isolated, source-verified adapter for fourth-persona lived nine-region learning.

This module deliberately does NOT modify daino's production task recorder.
It consumes only already-inserted, actual chosen task outcomes and checks
the source world event, grading, actor, time, action and task context.
"""
from __future__ import annotations
import math,json
import brain_9plus3_lived_learning as lived
import brain_9plus3_task_learning as task

GOAL_ACTIONS=("rest","wander","play")
ENGINES=lived.INTEGRATORS

def verified_ingest(c,profile,record_id):
    row=c.execute("""select o.id,o.profile,o.context_key,o.action,o.ts,
      o.task_success,o.predicted_prior,o.source_kind,o.event_id,
      e.id as world_verified,e.profile as world_profile,e.ts as world_ts,e.action as world_action
      from brain_9plus3_task_outcomes o
      join world_events e on e.id=o.event_id and e.profile=o.profile
      where o.id=?""",(int(record_id),)).fetchone()
    if row is None or row["profile"]!=profile or row["world_profile"]!=profile:
        raise ValueError("task source and world actor do not match")
    if abs(float(row["ts"])-float(row["world_ts"]))>1.:
        raise ValueError("task grade world event timestamp mismatch")
    if row["task_success"] not in (0,1):
        raise ValueError("not a binary verified grade")
    if profile=="sim_fourth":
        expected_action={"rest":"休息","wander":"探索","play":"遊び"}
        if expected_action.get(row["action"])!=row["world_action"]:
            raise ValueError("world action is not the task graded action")
    previous=None
    present=c.execute("""select 1 from sqlite_master
      where type='table' and name='brain_9plus3_lived_events'""").fetchone()
    if present:
        previous=c.execute("""select task_success from brain_9plus3_lived_events
          where source_id=?""",(task.TABLE+"#"+str(row["id"]),)).fetchone()
    if previous is not None and int(previous["task_success"])!=int(row["task_success"]):
        raise ValueError("outcome record was modified after ingestion")
    if profile=="sim_fourth":
        if row["source_kind"]!="simulation_environment":
            raise ValueError("synthetic source grading mismatch")
    elif row["source_kind"]!="verified_task_evaluator":
        raise ValueError("synthetic source cannot train real persona")
    return lived.observe_task(
        c,profile,row["context_key"],row["action"],row["ts"],
        task.TABLE+"#"+str(row["id"]),bool(row["task_success"]),
        row["predicted_prior"],source_kind=row["source_kind"])

def decide(c,profile,context,now,old_goal=None,disable=None,record=True):
    """Evidence-only, own-goal choice that can be ablated per integrator.

    This is a lab policy (not an LLM-authored personal aspiration).
    Goal candidates are action strategies available in the virtual world.
    """
    if disable not in (*ENGINES,None):
        raise ValueError("invalid ablation")
    history=task.evidence(c,profile,now,context,limit=128)
    snapshot=lived.snapshot(c,profile,context)
    p=task.posterior(history,GOAL_ACTIONS,now=now,half_life_days=4.)
    counts=p["counts"]
    estimates=p["expected_task_success"]
    total=sum(counts.values())
    middle=sum(counts.values())/len(counts)
    raw=snapshot["integrators"]
    def signal(key):
        if key==disable:return 0.
        return float(raw.get(key) if raw.get(key) is not None else .5)
    reward=signal("reward_value")
    world=signal("world_causal_model")
    reflect=signal("meta_reflection")
    scores={}
    for a in GOAL_ACTIONS:
        n=counts[a]
        confidence=n/(n+4.)
        exploitation=(estimates[a]-.5)*confidence*(.45+1.6*reward)
        uncertain=(1./math.sqrt(n+1.))*(.06+.65*(1.-world))
        reconsider=(.30*reflect if n<middle else -.05*reflect)
        stick=.11 if a==old_goal else 0.
        scores[a]=round(exploitation+uncertain+reconsider+stick,6)
    if total<6:
        chosen=old_goal if old_goal in GOAL_ACTIONS else GOAL_ACTIONS[0]
    else:
        chosen=max(GOAL_ACTIONS,key=lambda a:(scores[a],-GOAL_ACTIONS.index(a)))
    decision={"goal":chosen,"previous_goal":old_goal,
        "changed":old_goal is not None and chosen!=old_goal,
        "scores":scores,"engine_inputs":{"reward_value":reward,
         "world_causal_model":world,"meta_reflection":reflect},
        "observations":total,
        "sources":snapshot["sources"],
        "integrator_ablation":disable,
        "is_synthetic_personal_goal":True}
    if record and decision["changed"]:
        lived.ensure(c)
        c.execute("""insert into brain_9plus3_goal_choices(profile,context_key,
          chosen_goal,previous_goal,reason_json,source_ids_json,at,synthetic_model_review)
          values(?,?,?,?,?,?,?,1)""",(profile,context,chosen,old_goal,
            json.dumps(decision,ensure_ascii=False),
            json.dumps(decision["sources"]),now))
    return decision

def decide_probabilities(baseline,goal,*,weight=.42):
    """Keep all actions possible; an internal goal measurably affects choice."""
    if goal not in baseline:return dict(baseline)
    target={a:(.82 if a==goal else .09) for a in baseline}
    values={a:(1-weight)*baseline[a]+weight*target[a] for a in baseline}
    denominator=sum(values.values())
    return {a:v/denominator for a,v in values.items()}
