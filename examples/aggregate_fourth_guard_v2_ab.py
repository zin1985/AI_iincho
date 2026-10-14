"""Summarize paired live fourth-persona A/B for 4 shift scenarios.

Measures policy reward and expected regret separately. Sources are
independent sandbox JSON reports; no N memory, no LLM cost.
"""
from pathlib import Path
import json,argparse
BASE=Path(__file__).resolve().parents[1]/"reports"/"simulations"
SCENARIOS=("none","7","10","14")
def aggregate(seeds):
    rows=[]
    for seed in seeds:
        for scenario in SCENARIOS:
            f=BASE/f"fourth_guard_v2_{scenario}_{seed}"/"comparison.json"
            if not f.exists():continue
            doc=json.loads(f.read_text(encoding="utf8"))
            assert doc["all_results_synthetic"] and doc["N_persona_not_changed"]
            b=doc["baseline"]["metrics"];g=doc["guarded"]["metrics"]
            row={"seed":seed,"scenario":scenario,"source":str(f),
                 "changed_actions":doc["different_actions"],
                 "old_success":b["overall"]["successes"],
                 "new_success":g["overall"]["successes"],
                 "old_expected_regret":b["overall"]["mean_regret"],
                 "new_expected_regret":g["overall"]["mean_regret"],
                 "old_expected_success":b["overall"]["expected_success_rate"],
                 "new_expected_success":g["overall"]["expected_success_rate"],
                 "old_72h":b["first72"],"new_72h":g["first72"],
                 "suspended":doc["guarded"]["safety"]["suspended"],
                 "suspension_by_reason":doc["guarded"]["safety"],
                 "review_count":len(doc["guarded"]["strategy_reviews"])}
            rows.append(row)
    groups={}
    for scenario in SCENARIOS:
        subset=[r for r in rows if r["scenario"]==scenario]
        if not subset:continue
        denominator=432*len(subset)
        old=sum(r["old_success"] for r in subset)
        new=sum(r["new_success"] for r in subset)
        change=sum(r["changed_actions"] for r in subset)
        oldregret=sum(r["old_expected_regret"] for r in subset)/len(subset)
        newregret=sum(r["new_expected_regret"] for r in subset)/len(subset)
        comparison={"seeds":[r["seed"] for r in subset],
           "actions_each_arm":denominator,
           "old_success":old,"new_success":new,
           "old_success_rate_pct":round(100*old/denominator,3),
           "new_success_rate_pct":round(100*new/denominator,3),
           "delta_success_pp":round(100*(new-old)/denominator,3),
           "old_mean_regret":round(oldregret,5),
           "new_mean_regret":round(newregret,5),
           "regret_change":round(newregret-oldregret,5),
           "different_actions":change,
           "suppression_events":sum(r["suspended"] for r in subset)}
        if scenario!="none":
            before=sum(r["old_72h"]["successes"] for r in subset)
            after=sum(r["new_72h"]["successes"] for r in subset)
            n=sum(r["old_72h"]["n"] for r in subset)
            comparison["post_change_72h"]={
                "old_success":before,"new_success":after,"n":n,
                "old_pct":round(100*before/n,3),
                "new_pct":round(100*after/n,3)}
        groups[scenario]=comparison
    audit={"schema":"fourth_guard_v2_ab_audit_v1","seeds":seeds,
          "study":"18_day_same_initial_sqlite_same_oracle_no_llm",
          "safe_to_claim_production_improvement":False,
          "groups":groups,"rows":rows}
    dest=BASE/"fourth_guard_v2_study.json"
    dest.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"file":str(dest),"groups":groups},ensure_ascii=False))
    return audit

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[941])
    a=p.parse_args();aggregate(a.seeds)