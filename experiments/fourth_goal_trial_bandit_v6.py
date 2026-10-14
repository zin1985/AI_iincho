"""Sandbox-only bounded daily test actions for nonstationary nine-three goals.

The three integrators still choose a provisional goal. A separate empirical
tester may request at most ONE alternative action per 24 virtual hours in a
given task, and only when grade-backed deterioration and alternative support
are present. A request is not a success claim and is only a scheduling intent.
"""
from __future__ import annotations
from tools import fourth_goal_trial_bandit_v4 as detector
from tools import fourth_lived_pipeline as original
from tools.fourth_goal_adoption_gate_v2 import _unit
DAY=86400.
def gate(c,profile,now,context,prob,goal,*,max_weight=.42):
    if profile!="sim_fourth":
        raise ValueError("v6 is experiment-four only")
    standard=original.decide_probabilities(prob,goal,weight=max_weight)
    candidate=detector.gate(c,profile,now,context,_unit(prob),goal)
    alternate=candidate.get("effective_goal")
    rates=candidate.get("recent_estimates",{})
    counts=candidate.get("counts",{})
    can_trial=(candidate.get("reason")=="trial_alternative_after_verified_decline" and
          alternate!=goal and counts.get(alternate,0)>=5 and
          rates.get(alternate,0.)>=rates.get(goal,0.)+.15)
    if can_trial:
        c.execute("""create table if not exists brain_9plus3_trial_intents(
          id integer primary key,profile text not null,
          context_key text not null,candidate text not null,
          at real not null,source_ids_json text not null)""")
        latest=c.execute("""select max(at) from brain_9plus3_trial_intents
          where profile=? and context_key=?""",(profile,context)).fetchone()[0]
        if latest is None or float(now)-float(latest)>=DAY:
            import json
            c.execute("""insert into brain_9plus3_trial_intents
                (profile,context_key,candidate,at,source_ids_json)
                values(?,?,?,?,?)""",(profile,context,alternate,float(now),
                    json.dumps(candidate.get("sources",[]))))
            # This is an experimental request, not the outcome; the regular
            # world-event and independent grade will say what actually happened.
            weights={a:(1.0 if a==alternate else 0.0) for a in standard}
            reason="daily_bounded_alternative_trial_requested"
            trial=alternate
        else:
            weights=standard
            reason="daily_trial_budget_spent"
            trial=None
    else:
        weights=standard
        reason="keep_stable_original_goal"
        trial=None
    return {"weights":weights,"reason":reason,"strength":float(trial is not None),
       "effective_goal":goal,"changed_goal":False,"previous_goal":goal,
       "sources":candidate.get("sources",[]),
       "source_ids":candidate.get("sources",[]),
       "observations":sum(counts.values()),
       "trial_action_requested":trial,
       "max_trials_per_task_per_24h":1,
       "objective_reward_not_inferred":True}
