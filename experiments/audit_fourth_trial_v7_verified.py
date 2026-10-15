"""Read-only audit for multi-seed v7 nine-plus-three verified probe trials.

Checks intents against actual executed world events and source-linked graded
outcomes; estimates paired regret and success differences. The oracle is
used post hoc only, never for decisions.
"""
from __future__ import annotations
import argparse,json,sqlite3,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/"reports/simulations"
LABEL={"rest":"休息","wander":"探索","play":"遊び"}
def read(path):
    return json.loads(path.read_text(encoding="utf8"))
def trial_audit(arm):
    p=Path(arm["source_db"])
    with sqlite3.connect("file:"+p.as_posix()+"?mode=ro",uri=True) as c:
        c.row_factory=sqlite3.Row
        schema={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
        if "brain_9plus3_trial_intents" not in schema:
            return {"requested":0,"executed":0,"graded":0,"successes":0,"not_executed":0,"details":[]}
        intents=c.execute("select id,context_key,candidate,at from brain_9plus3_trial_intents order by at,id").fetchall()
        details=[]
        for intent in intents:
            evt=c.execute("""select id,action from world_events
               where ts=? and profile='sim_fourth' order by id desc limit 1""",
               (float(intent["at"]),)).fetchone()
            did=bool(evt and evt["action"]==LABEL[intent["candidate"]])
            grade=c.execute("""select id,task_success from brain_9plus3_task_outcomes
              where event_id=? and profile='sim_fourth' and context_key=?
              and action=? and source_kind='simulation_environment'""",
              (evt["id"] if evt is not None else -1,intent["context_key"],intent["candidate"])).fetchone() if did else None
            if grade is not None:
                brain=c.execute("""select region_json,integrators_json from
                   brain_9plus3_lived_events where profile='sim_fourth'
                   and source_id=?""",("brain_9plus3_task_outcomes#"+str(grade["id"]),)).fetchone()
                assert brain is not None,"trial outcome not fed back into nine-region ledger"
                assert len(json.loads(brain["region_json"]))==9
                assert len(json.loads(brain["integrators_json"]))==3
            details.append({"step":int((float(intent["at"])-1792040000.)//3600.),
                "candidate":intent["candidate"],"executed":did,
                "graded":grade is not None,
                "fed_into_all_nine_and_three":grade is not None,
                "success":bool(grade["task_success"]) if grade is not None else None,
                "grade_id":int(grade["id"]) if grade is not None else None})
        return {"requested":len(details),"executed":sum(x["executed"] for x in details),
                "graded":sum(x["graded"] for x in details),
                "successes":sum(x["success"] is True for x in details),
                "not_executed":sum(not x["executed"] for x in details),
                "details":details}

def run(seeds=(978,979,980),days=18,shift_cases=("9","none")):
    samples=[];groups={}
    for scene in shift_cases:
        entries=[]
        for seed in seeds:
            folder=BASE/f"fourth_lived_trial_bandit_v7_{days}d_{seed}_{scene}"
            old=folder/"full.json"; new=folder/"full_v2.json"
            if not old.is_file() or not new.is_file():
                print("MISSING",seed,scene,flush=True)
                continue
            a,b=read(old),read(new)
            assert len(a["trace"])==days*24==len(b["trace"])
            assert b["verified_region_ingestions"]==days*24
            assert b["independent_persona_rows"]==0
            proof=trial_audit(b)
            assert proof["requested"]==proof["graded"],"Unexecuted scheduled trial!"
            assert proof["requested"]==proof["executed"]
            diffs=sum(u["action"]!=v["action"] for u,v in zip(a["trace"],b["trace"]))
            regret_delta=a["oracle_mean_regret"]-b["oracle_mean_regret"]
            start=int(scene) if scene!="none" else None
            def w(trace):
                part=[r for r in trace if start is not None and start<=r["day"]<start+3]
                return sum(r["success"] for r in part) if part else None
            rec={"seed":seed,"scenario":scene,
              "success_old":a["successes"],"success_new":b["successes"],
              "success_delta":b["successes"]-a["successes"],
              "regret_old":a["oracle_mean_regret"],
              "regret_new":b["oracle_mean_regret"],
              "regret_improvement":round(regret_delta,5),
              "action_differences":diffs,
              "goal_differences":sum(x["goal"]!=y["goal"] for x,y in zip(a["trace"],b["trace"])),
              "first72_old":w(a["trace"]),"first72_new":w(b["trace"]),
              "nine_three_events":b["verified_region_ingestions"],
              "trial_provenance":proof,"gate_reasons":b["gate_reasons"]}
            entries.append(rec);samples.append(rec)
        if entries:
            groups[scene]={"n_seeds":len(entries),
                "n_actions_each_arm":days*24*len(entries),
                "success_old":sum(x["success_old"] for x in entries),
                "success_new":sum(x["success_new"] for x in entries),
                "delta_success":sum(x["success_delta"] for x in entries),
                "mean_regret_improvement":round(statistics.mean(x["regret_improvement"] for x in entries),5),
                "all_action_changes":sum(x["action_differences"] for x in entries),
                "trial_requests":sum(x["trial_provenance"]["requested"] for x in entries),
                "confirmed_executions":sum(x["trial_provenance"]["executed"] for x in entries),
                "trial_success":sum(x["trial_provenance"]["successes"] for x in entries),
                "first72_old":sum(x["first72_old"] or 0 for x in entries),
                "first72_new":sum(x["first72_new"] or 0 for x in entries)}
    report={"schema":"fourth_v7_verified_exploration_audit",
      "days":days,"samples":samples,"conditions":groups,
      "provenance_checked":True,"real_N_changed":False,
      "synthetic_goals_not_personal_values":True}
    output=BASE/"fourth_v7_verified_exploration_audit.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"file":str(output),"conditions":groups},ensure_ascii=False),flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seeds",nargs="+",type=int,default=[978,979,980])
    p.add_argument("--days",type=int,default=18)
    a=p.parse_args()
    run(a.seeds,a.days)
