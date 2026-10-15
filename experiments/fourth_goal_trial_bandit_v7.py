"""Safe causal probe v7 for isolated fourth-persona nine-plus-three.

Unlike v6, a requested probe must be executed by the simulator explicitly.
Only actual, independently graded prior probe actions count toward the
stop/continue rule. The original 9+3 policy remains byte-for-byte the
fallback, and no user persona, goals or autobiographical memory is changed.
"""
from __future__ import annotations
from tools import fourth_goal_trial_bandit_v6 as v6

COOLDOWN=3*86400.
def _verified_trials(c,profile,context,now):
    exists=c.execute("""select 1 from sqlite_master
        where type='table' and name='brain_9plus3_trial_intents'""").fetchone()
    if not exists:return []
    # The source ledger is the authority. An intent without a chosen world
    # event and independent objective grade NEVER counts as a trial success.
    rows=c.execute("""select t.id,t.candidate,t.at,
        o.task_success as passed,o.id as outcome_id
        from brain_9plus3_trial_intents t
        left join world_events w on w.profile=t.profile and w.ts=t.at
          and w.action=case t.candidate
             when 'rest' then '休息' when 'wander' then '探索'
             when 'play' then '遊び' end
        left join brain_9plus3_task_outcomes o
          on o.event_id=w.id and o.profile=t.profile
          and o.context_key=t.context_key and o.action=t.candidate
          and o.source_kind='simulation_environment'
        where t.profile=? and t.context_key=? and t.at<?
        order by t.at desc,t.id desc limit 12""",
       (profile,context,float(now))).fetchall()
    return [{"id":int(r["id"]),"candidate":r["candidate"],
       "at":float(r["at"]),"passed":(bool(r["passed"]) if r["passed"] is not None else None),
       "outcome_id":r["outcome_id"]} for r in rows]

def gate(c,profile,now,context,prob,goal,*,max_weight=.42):
    if profile!="sim_fourth":raise ValueError("synthetic trial gate only")
    prior=_verified_trials(c,profile,context,now)
    # Two consecutive *executed and failed* probes of the same alternative:
    # stop forcing more; respect existing 9+3. The stop is local to a task
    # and does not edit any real persona's goals or memory.
    graded=[r for r in prior if r["passed"] is not None]
    cooldown=bool(len(graded)>=2 and
       graded[0]["candidate"]==graded[1]["candidate"] and
       not graded[0]["passed"] and not graded[1]["passed"] and
       now-graded[0]["at"]<COOLDOWN)
    if cooldown:
        from tools import fourth_lived_pipeline as pipe
        return {"weights":pipe.decide_probabilities(prob,goal,weight=max_weight),
            "reason":"verified_probe_failures_cooldown","strength":0.,
            "effective_goal":goal,"changed_goal":False,"previous_goal":goal,
            "sources":["brain_9plus3_task_outcomes#"+str(x["outcome_id"])
                       for x in graded[:2]],
            "source_ids":["brain_9plus3_task_outcomes#"+str(x["outcome_id"])
                       for x in graded[:2]],"observations":0,
            "trial_action_requested":None,"trial_outcomes":[0,0],
            "causal_probe_executed":False}
    result=v6.gate(c,profile,now,context,prob,goal,max_weight=max_weight)
    result["trial_action_requested"]=result.get("trial_action_requested")
    result["trial_outcomes"]=[int(x["passed"]) for x in graded[:5]]
    result["causal_probe_executed"]=bool(result["trial_action_requested"])
    return result
