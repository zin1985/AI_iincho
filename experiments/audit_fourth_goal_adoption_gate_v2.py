"""Paired sandbox audit of outcome-gated 9+3 goal influence.

Read-only experiment outputs; the hidden success probabilities are used only
for post-hoc scoring, never to steer the agent.
"""
import json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/simulations"
def summarize(seeds,scenarios,days):
    records=[]
    for scenario in scenarios:
        for seed in seeds:
            directory=OUT/f"fourth_lived_gate_v2_{days}d_{seed}_{scenario}"
            old_file=directory/"full.json"
            new_file=directory/"full_v2.json"
            if not (old_file.is_file() and new_file.is_file()):
                print("NOT_READY",seed,scenario,flush=True)
                continue
            old=json.loads(old_file.read_text(encoding="utf8"))
            new=json.loads(new_file.read_text(encoding="utf8"))
            assert old["n"]==days*24==new["n"]
            assert old["verified_region_ingestions"]==new["verified_region_ingestions"]==days*24
            assert old["independent_persona_rows"]==new["independent_persona_rows"]==0
            before=old["trace"];after=new["trace"]
            changes=sum(a["action"]!=b["action"] for a,b in zip(before,after))
            goals=sum(a["goal"]!=b["goal"] for a,b in zip(before,after))
            shift=None if scenario=="none" else int(scenario)
            def window(rows):
                return {"n":len(rows),"success":sum(bool(x["success"]) for x in rows),
                    "expected_regret":round(sum(x["oracle_regret"] for x in rows)/len(rows),5)}
            old72=window([x for x in before if shift is not None and shift<=x["day"]<shift+3]) if shift is not None else None
            new72=window([x for x in after if shift is not None and shift<=x["day"]<shift+3]) if shift is not None else None
            records.append({"seed":seed,"scenario":scenario,"source":str(directory),
              "baseline_success":old["successes"],"new_success":new["successes"],
              "change_in_success":new["successes"]-old["successes"],
              "baseline_regret":old["oracle_mean_regret"],"new_regret":new["oracle_mean_regret"],
              "regret_improvement":round(old["oracle_mean_regret"]-new["oracle_mean_regret"],5),
              "changed_actions":changes,"changed_goals":goals,
              "old_goals_changed":old["goal_change_count"],
              "new_goals_changed":new["goal_change_count"],
              "gate_reasons":new["gate_reasons"],
              "old72":old72,"new72":new72})
    groups={}
    for scenario in scenarios:
        arr=[r for r in records if r["scenario"]==scenario]
        if not arr:continue
        n=days*24*len(arr)
        groups[scenario]={"seeds":[r["seed"] for r in arr],
          "n_each_arm":n,
          "baseline_successes":sum(r["baseline_success"] for r in arr),
          "new_successes":sum(r["new_success"] for r in arr),
          "delta_successes":sum(r["change_in_success"] for r in arr),
          "mean_regret_improvement":round(sum(r["regret_improvement"] for r in arr)/len(arr),5),
          "action_changes":sum(r["changed_actions"] for r in arr),
          "goal_hours_different":sum(r["changed_goals"] for r in arr),
          "gate_activation_counts":{k:sum(r["gate_reasons"].get(k,0) for r in arr)
              for k in ("observed_goal_support","goal_advantage_unproven",
                  "goal_underperforms_existing_policy","insufficient_observed_goal_evidence")}}
        if scenario!="none":
            groups[scenario]["first72_old"]=sum(r["old72"]["success"] for r in arr)
            groups[scenario]["first72_new"]=sum(r["new72"]["success"] for r in arr)
    report={"schema":"fourth_goal_adoption_gate_v2_paired","days":days,
      "records":records,"groups":groups,
      "theoretical_emotions_not_asserted":True,
      "objective_hidden_oracle_never_in_training":True,
      "production_N_changes":0}
    p=OUT/"fourth_goal_adoption_gate_v2_study.json"
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"report":str(p),"groups":groups},ensure_ascii=False),flush=True)
    return report
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,nargs="+",default=[972,973,974])
    parser.add_argument("--scenarios",nargs="+",default=["9","none"])
    parser.add_argument("--days",type=int,default=18)
    a=parser.parse_args()
    summarize(a.seeds,a.scenarios,a.days)
