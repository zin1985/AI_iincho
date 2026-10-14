"""Read-only detailed audit for exact-initial-world four-arm experiments."""
from __future__ import annotations
import json,math,argparse,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"reports"/"simulations"
STAGES=(
 {"key":"material_observation","goal":"素材を観察","p":{"rest":.18,"wander":.38,"play":.82}},
 {"key":"playful_experiment","goal":"遊びから発見","p":{"rest":.16,"wander":.83,"play":.32}},
 {"key":"recovery_and_revision","goal":"休んで組み直す","p":{"rest":.81,"wander":.36,"play":.20}})
ACTIONS=("rest","wander","play")
NAMES=("no_9plus3","outcome_9plus3","active_9plus3","active_with_review")

def observe(arm,days):
    trace=arm["trace"];assert len(trace)==days*24
    rows=[]
    for i,item in enumerate(trace):
        assert item["step"]==i
        stage=STAGES[item["phase"]]
        p=stage["p"]
        if item["day"]>=days//2:
            p={a:stage["p"][ACTIONS[(ACTIONS.index(a)+1)%3]] for a in ACTIONS}
        assert item["optimal"]==max(p,key=p.get)
        rows.append({"day":item["day"],"phase":item["phase"],"action":item["chosen"],
           "passed":int(item["passed"]),"optimal":int(item["optimal"]==item["chosen"]),
           "regret":max(p.values())-p[item["chosen"]],
           "expected_p":p[item["chosen"]],
           "calibration":(item["model_prior"]-int(item["passed"]))**2})
    def summarize(sub):
        n=len(sub)
        if not n:return {"n":0}
        counts={a:sum(q["action"]==a for q in sub) for a in ACTIONS}
        ent=-sum((v/n)*math.log2(v/n) for v in counts.values() if v)
        return {"n":n,"successful":sum(q["passed"] for q in sub),
           "success_pct":round(100*sum(q["passed"] for q in sub)/n,3),
           "oracle_expected_success_pct":round(100*sum(q["expected_p"] for q in sub)/n,3),
           "regret":round(sum(q["regret"] for q in sub)/n,5),
           "optimal_pct":round(100*sum(q["optimal"] for q in sub)/n,3),
           "brier":round(sum(q["calibration"] for q in sub)/n,5),
           "action_counts":counts,"entropy_bits":round(ent,3)}
    return {"total":summarize(rows),
      "before_drift":summarize([x for x in rows if x["day"]<days//2]),
      "after_drift":summarize([x for x in rows if x["day"]>=days//2]),
      "first_72h_drift":summarize([x for x in rows if days//2<=x["day"]<days//2+3]),
      "late_drift":summarize([x for x in rows if x["day"]>=days//2+3]),
      "per_goal":{x["key"]:summarize([r for r in rows if r["phase"]==i])
                   for i,x in enumerate(STAGES)},
      "day_by_day":[{"day":day,**summarize([r for r in rows if r["day"]==day])}
                   for day in range(days)]}

def go(seeds,days):
    trials=[];missing=[]
    for seed in seeds:
        f=BASE/f"fourth_controlled_{days}d_{seed}"/"controlled_report.json"
        if not f.exists():
            missing.append(seed);continue
        doc=json.loads(f.read_text(encoding="utf8"))
        if not doc.get("no_other_persona_events") or not doc.get("same_hidden_task_oracle"):
            raise ValueError("invalid control marker")
        if set(doc["arms"])!=set(NAMES):raise ValueError("incomplete 4-arm study")
        fingerprints={doc["arms"][name]["starting_snapshot_sha256"] for name in NAMES}
        assert len(fingerprints)==1
        arms={}
        for name in NAMES:
            v=doc["arms"][name]["data"]
            assert v["objective_task_outcomes"]==days*24
            assert v["unrelated_persona_events"]==0
            arms[name]={"metrics":observe(v,days),
                "approved_strategies":doc["arms"][name]["strategy_count"],
                "review_log":v.get("strategy_review_log",[]),
                "micro_nine_three_feedback":v["micro_nine_three_feedback"],
                "reflection_notes":[{"day":x["day"],
                  "hypothesis":x["brain_evidence"]["reflective_note"]["hypothesis"],
                  "observation":x["brain_evidence"]["reflective_note"]["observation"],
                  "reason_to_review":x["brain_evidence"]["reflective_note"]["reason_to_review"],
                  "sources":x["brain_evidence"]["reflective_note"]["verified_sources"][:3],
                  "candidate":x["brain_evidence"]["reflective_note"]["potential_action"],
                  "objective_not_private_introspection":True}
                  for x in v.get("cognitive_9plus3_diary",[])],
                "trace":v["trace"]}
        deltas=[]
        for first,second in (("no_9plus3","outcome_9plus3"),
                             ("outcome_9plus3","active_9plus3"),
                             ("active_9plus3","active_with_review")):
            a,b=arms[first],arms[second]
            changed=sum(x["chosen"]!=y["chosen"] for x,y in zip(a["trace"],b["trace"]))
            deltas.append({"from":first,"to":second,"changed_actions":changed,
                "success_diff":b["metrics"]["total"]["successful"]-a["metrics"]["total"]["successful"],
                "regret_diff":round(b["metrics"]["total"]["regret"]-
                                    a["metrics"]["total"]["regret"],5),
                "after_drift_success_diff":b["metrics"]["after_drift"]["successful"]-
                                            a["metrics"]["after_drift"]["successful"]})
        trials.append({"seed":seed,"source":str(f),
            "days":days,"sha256_identical_at_start":True,"arms":arms,
            "ablations":deltas})
    # Aggregate comparable numerators, not per-run rounded rates.
    pooled={}
    for name in NAMES:
        samples=[t["arms"][name] for t in trials]
        if not samples:continue
        n=sum(a["metrics"]["total"]["n"] for a in samples)
        successes=sum(a["metrics"]["total"]["successful"] for a in samples)
        pooled[name]={"seeds":len(samples),"actions":n,
             "successes":successes,"success_pct":round(100*successes/n,3),
             "average_expected_regret":round(sum(a["metrics"]["total"]["regret"]*
                     a["metrics"]["total"]["n"] for a in samples)/n,5),
             "first72_successes":sum(a["metrics"]["first_72h_drift"]["successful"] for a in samples),
             "first72_actions":sum(a["metrics"]["first_72h_drift"]["n"] for a in samples),
             "total_reviews":sum(len(a["review_log"]) for a in samples)}
    individual_review_effect=[]
    for t in trials:
        a=t["arms"]["active_9plus3"]["metrics"]["total"]
        b=t["arms"]["active_with_review"]["metrics"]["total"]
        individual_review_effect.append({"seed":t["seed"],"review_adoptions":
             len(t["arms"]["active_with_review"]["review_log"]),
             "reward_difference":b["successful"]-a["successful"],
             "regret_difference":round(b["regret"]-a["regret"],5),
             "different_actions":t["ablations"][-1]["changed_actions"]})
    # Check source records never used to train a real persona.
    out={"schema":"controlled_fourth_4arm_audit_v1",
       "seed_set":seeds,"complete_seeds":[x["seed"] for x in trials],
       "missing_seeds":missing,"days":days,
       "synthetic_only":True,
       "no_external_model_calls":True,
       "oracle_used_for_audit_only":True,
       "no_personality_consciousness_proof":True,
       "arms_pooled":pooled,"self_revision_effects":individual_review_effect,
       "trial_details":trials}
    folder=BASE/f"fourth_controlled_{days}d_summary"
    folder.mkdir(exist_ok=True)
    dest=folder/"audit.json"
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"path":str(dest),"pooled":pooled,
        "review_effect":individual_review_effect,
        "missing_seeds":missing},ensure_ascii=False))
    return out
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--seeds",type=int,nargs="+",default=[921,922,923])
 p.add_argument("--days",type=int,default=18)
 args=p.parse_args();go(args.seeds,args.days)