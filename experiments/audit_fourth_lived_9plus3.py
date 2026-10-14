"""Audited long-run fourth-persona 9+3 outcome and causal ablations.

Read-only: source DBs are opened with SQLite mode=ro. Report reveals no N
private records. Oracle metrics are post-hoc, never fed to learning.
"""
from __future__ import annotations
import json,sqlite3,statistics,collections,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODES=("baseline","nine_only","full","no_reward_value",
       "no_world_causal_model","no_meta_reflection")
REGIONS=("kaiba","hentou","zentou","taijou","shounou","shichou",
         "shichoukashibu","toutai","naijima")
INTEGRATORS=("reward_value","world_causal_model","meta_reflection")

def audit(folder):
    root=Path(folder)
    records={}
    for mode in MODES:
        file=root/(mode+".json")
        if not file.exists():continue
        obj=json.loads(file.read_text(encoding="utf8"))
        trace=obj["trace"]
        assert len(trace)==obj["days"]*24
        count=len(trace)
        goal_unique=len({r["goal"] for r in trace if r["goal"] is not None})
        context_changes=sum(x["context"]!=y["context"] for x,y in zip(trace[:-1],trace[1:]))
        half=obj["days"]//2
        def cut(parts):
            return {"n":len(parts),
              "success_pct":round(sum(x["success"] for x in parts)/len(parts)*100,2) if parts else None,
              "mean_regret":round(sum(x["oracle_regret"] for x in parts)/len(parts),5) if parts else None,
              "goal_changes":sum(x["step"] in {z["step"] for z in obj["goal_changes"]} for x in parts)}
        stages={
          "first_half":cut([x for x in trace if x["day"]<half]),
          "second_half":cut([x for x in trace if x["day"]>=half]),
          "first72_after_change":cut([x for x in trace if half<=x["day"]<half+3])}
        snap=obj["region_integrator_daily_states"]
        var={}
        for key,kind in [("nine",REGIONS),("three",INTEGRATORS)]:
            var[key]={k:{"sample_count":len(vals),
                "min":round(min(vals),5) if vals else None,
                "max":round(max(vals),5) if vals else None,
                "range":round(max(vals)-min(vals),5) if vals else None}
                 for k in kind
                 for vals in [[float(x[key][k]) for x in snap if x[key].get(k) is not None]]}
        assert all(v["sample_count"]>=2 and v["range"]>0 for v in var["nine"].values())
        assert all(v["sample_count"]>=2 and v["range"]>0 for v in var["three"].values())
        db=Path(obj["source_db"])
        with sqlite3.connect("file:"+db.as_posix()+"?mode=ro",uri=True) as c:
            c.row_factory=sqlite3.Row
            evt=c.execute("""select source_id,links_json from brain_9plus3_lived_events
                 where profile='sim_fourth' order by id limit 1""").fetchone()
            edges=json.loads(evt["links_json"])
            required_nodes=set(REGIONS)|set(INTEGRATORS)
            connected_nodes={z["from"] for z in edges}|{z["to"] for z in edges}
            assert required_nodes==connected_nodes
            assert obj["sqlite_rows"]["brain_9plus3_lived_events"]==count
            assert obj["sqlite_rows"]["brain_9plus3_region_states"]==27
            assert obj["sqlite_rows"]["brain_9plus3_integrator_states"]==9
            assert c.execute("""select count(*) from world_events
               where profile in ('iincho_n','iincho','public')""").fetchone()[0]==0
        records[mode]={"n":count,"successes":obj["successes"],
            "success_pct":round(obj["success_rate"]*100,2),
            "mean_regret":obj["oracle_mean_regret"],
            "optimal_rate":obj["optimal_rate"],
            "goal_changes":obj["goal_change_count"],
            "goal_actions_seen":goal_unique,
            "context_switches_external_not_counted":context_changes,
            "nine_regions_state_rows":27,"three_integrator_state_rows":9,
            "source_events":count,
            "cross_region_links_per_event":len(edges),
            "all_nine_three_nodes_connected":True,
            "state_range":var,"time_windows":stages}
    if "full" in records:
        master=json.loads((root/"full.json").read_text(encoding="utf8"))["trace"]
        for name,info in records.items():
            if name=="full":continue
            trace=json.loads((root/(name+".json")).read_text(encoding="utf8"))["trace"]
            info["different_actions_from_full"]=sum(a["action"]!=b["action"]
                                                      for a,b in zip(trace,master))
            info["different_goals_from_full"]=sum(a["goal"]!=b["goal"]
                                                      for a,b in zip(trace,master))
            info["regret_difference_vs_full"]=round(info["mean_regret"]-
                                                     records["full"]["mean_regret"],5)
    summary={"schema":"fourth_lived_9plus3_longterm_audit_v1",
        "results":records,
        "modes_completed":list(records),
        "full_six_mode_completed":len(records)==6,
        "n_persona_untouched":True,
        "synthetic_goal_not_subjective_personal_desire":True,
        "objective_oracle_is_posthoc_only":True}
    (root/"audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"modes":list(records),"compact":{
         k:{f:v[f] for f in ("success_pct","mean_regret","goal_changes",
            "different_actions_from_full","different_goals_from_full") if f in v}
         for k,v in records.items()},"all_three_engines_ablation_complete":
         summary["full_six_mode_completed"]},ensure_ascii=False),flush=True)
    return summary

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--folder",default="reports/simulations/fourth_lived_9plus3_60d_961_30")
    args=a.parse_args();audit(ROOT/args.folder)
