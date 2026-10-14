"""9+3 experimental cognitive regulation and *inspectable* reflection.

This is the transparent observable state of a rule-based simulated cognition,
not an LLM's private thoughts and not the persona's subjective consciousness.
The outputs only use verified task outcomes with source event IDs, and are
read-only. The online policy intervention remains simulation-only until
verified across more tasks and seeds.
"""
from __future__ import annotations
from collections import Counter
from math import sqrt
import brain_9plus3_task_learning as task

REGIONS=("kaiba","hentou","zentou","taijou","shounou","shichou",
         "shichoukashibu","toutai","naijima")
INTEGRATORS=("reward_value","world_causal_model","meta_reflection")

def inspect(c,profile,at,task_context,goal,action_weights,*,last_n=30):
    history=task.evidence(c,profile,at,task_context,limit=128)
    recent=history[:last_n]
    recent12=history[:12]
    counts=Counter(x["action"] for x in recent)
    fails=[x for x in recent if not x["passed"]]
    total=len(history)
    successful=sum(x["passed"] for x in recent12)
    recent_rate=(successful/len(recent12)) if recent12 else None
    old12=history[12:24]
    old_rate=sum(x["passed"] for x in old12)/len(old12) if old12 else None
    errors=[abs(float(x["passed"])-float(x["prior_success_prediction"]))
            for x in recent12 if x["prior_success_prediction"] is not None]
    prediction_error=sum(errors)/len(errors) if errors else None
    post=task.posterior(history,list(action_weights))
    candidates=post["expected_task_success"]
    best=max(candidates,key=candidates.get) if candidates and total else None
    habit=counts.most_common(1)[0][0] if counts else None
    conflict=(best is not None and habit is not None and habit!=best)
    uncertainty={a:round(sqrt(max(0.0,p*(1-p)/(post["counts"][a]+3))),4)
                 for a,p in candidates.items()}
    novelty={a:post["counts"][a] for a in candidates}
    surprise=(old_rate is not None and recent_rate is not None and
              old_rate-recent_rate>=.20 and len(recent12)>=8)
    source_ids=[r["source"] for r in recent12]
    # The nine role-specific processors do different observable work.
    regions={
      "kaiba":{"retrieved_events":total,"recent_source_ids":source_ids[:4],
               "source":"verified_simulated_task_outcomes"},
      "hentou":{"objective_salience":"possible_change" if surprise else
                 "failure_review" if len(fails)>=6 else "normal",
                "felt_emotion":None},
      "zentou":{"current_goal":str(goal)[:120],
                 "best_evidenced_action":best,
                 "strategy_conflict":bool(conflict)},
      "taijou":{"habit_action":habit,"recent_action_counts":dict(counts),
                 "habit_is_not_personality":True},
      "shounou":{"prediction_error_mean":round(prediction_error,4)
                 if prediction_error is not None else None,
                 "estimated_success":{k:round(v,3) for k,v in candidates.items()}},
      "shichou":{"attention_target":"unexpected_decline" if surprise else
                 "strategy_conflict" if conflict else
                 "uncertain_untried_action" if min(novelty.values(),default=0)<3
                 else "current_goal"},
      "shichoukashibu":{"recent_activity_count":len(recent),
                 "physical_fatigue":None,"rest_not_forced":True},
      "toutai":{"review_triggered":surprise or conflict,
                 "evidence_for_method_change":bool(surprise or conflict),
                 "N_authored_reflection":False},
      "naijima":{"felt_satisfaction":None,"subjective_state":None,
                 "objective_state_only":True}}
    assert set(regions)==set(REGIONS)
    engines={
      "reward_value":{"outcome_window_n":len(recent12),
                      "window_success_rate":round(recent_rate,4) if recent_rate is not None else None,
                      "intrinsic_felt_reward":None},
      "world_causal_model":{"before_success_rate":round(old_rate,4)
                            if old_rate is not None else None,
                            "recent_success_rate":round(recent_rate,4) if recent_rate is not None else None,
                            "possible_environment_change":bool(surprise),
                            "not_a_causal_proof":True},
      "meta_reflection":{"habit_vs_evidence_conflict":bool(conflict),
                         "should_reconsider":bool(surprise or conflict),
                         "candidate_next_action":best,
                         "external_evidence_count":total,
                         "not_N_personal_diary":True}}
    assert set(engines)==set(INTEGRATORS)
    # Explore when little is known or when measured outcomes conflict with
    # an established strategy. Avoid forcing aimless activity at high confidence.
    if total<6:
        exploration=.30
    elif surprise:
        exploration=.28
    elif conflict:
        exploration=.22
    elif min(novelty.values(),default=0)<4:
        exploration=.17
    elif total>=18 and prediction_error is not None and prediction_error<.42:
        exploration=.07
    else:
        exploration=.13
    # A machine-readable consideration, not an unverifiable thought.
    if surprise:
        hypothesis="最近の成果が以前より下がった。環境変化を仮説として検証し、以前の方法に固執しない。"
    elif conflict:
        hypothesis="習慣になった行動と、観測上有効な行動に食い違いがある。別の方法を比較する。"
    elif total<6:
        hypothesis="まだ採点済みの経験が少ない。未確認の手法も慎重に試して情報を増やす。"
    else:
        hypothesis="現在の課題で有効そうな行動を継続し、次の失敗や変化を見落とさない。"
    reflection={
      "kind":"simulation_evidence_hypothesis_not_private_thought",
      "goal":str(goal)[:120],
      "observed":total,
      "verified_sources":source_ids[:5],
      "observation":("直近"+str(len(recent12))+"件の課題成功率 "+
            str(round(recent_rate*100,1))+"%。"
            if recent_rate is not None else "採点済みの経験がまだないため、成功率は未評価。"),
      "prediction_error":regions["shounou"]["prediction_error_mean"],
      "hypothesis":hypothesis,
      "potential_action":best,
      "habit_action":habit,
      "reason_to_review":regions["shichou"]["attention_target"],
      "machine_preference_not_personal_feeling":True,
      "unobserved_subjective_feelings":None}
    return {"nine_regions":regions,"three_integrators":engines,
            "reflective_note":reflection,
            "exploration_epsilon":exploration,"context":task_context,
            "has_real_N_inner_dialogue":False,"source_count":total}
