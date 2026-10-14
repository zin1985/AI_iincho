"""Paired 4th-persona online A/B: historic policy vs v2 evidence guard.

Never writes to N's DB. Both arms copy one canonical simulated-world SQLite.
Only evaluated outcomes from selected actions enter 9+3 training.
"""
from __future__ import annotations
import argparse,os,hashlib,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import sim_fourth_graded_outcomes as sim
import brain_9plus3_self_revision as revision
import brain_9plus3_strategy_guard_v2 as guard

def assess_trace(trace,shift_day):
    rows=[]
    for t in trace:
        day=t["day"]
        stage=sim.STAGES[t["phase"]]
        q=stage["quality"]
        if shift_day is not None and day>=shift_day:
            q={a:stage["quality"][sim.ACTIONS[(sim.ACTIONS.index(a)+1)%3]]
               for a in sim.ACTIONS}
        best=max(q.values())
        assert t["optimal"]==max(q,key=q.get)
        rows.append({"day":day,"success":int(t["passed"]),
                     "expected_p":q[t["chosen"]],
                     "regret":best-q[t["chosen"]]})
    def aggregate(rows):
        if not rows:return None
        return {"n":len(rows),"successes":sum(x["success"] for x in rows),
            "success_rate":round(sum(x["success"] for x in rows)/len(rows),5),
            "expected_success_rate":round(sum(x["expected_p"] for x in rows)/len(rows),5),
            "mean_regret":round(sum(x["regret"] for x in rows)/len(rows),5)}
    return {"overall":aggregate(rows),
       "first72":(aggregate([r for r in rows if shift_day<=r["day"]<shift_day+3])
                  if shift_day is not None else None),
       "after":(aggregate([r for r in rows if r["day"]>=shift_day])
                if shift_day is not None else None),
       "before":aggregate([r for r in rows if shift_day is None or r["day"]<shift_day])}

def run(seed,scenario):
    assert scenario in ("none","7","10","14")
    assert sim.DAYS==18,sim.DAYS
    shift_day=None if scenario=="none" else int(scenario)
    target=ROOT/"reports"/"simulations"/f"fourth_guard_v2_{scenario}_{seed}"
    if (target/"comparison.json").exists():
        obj=json.loads((target/"comparison.json").read_text(encoding="utf8"))
        print(json.dumps({"reused":True,"result":str(target/"comparison.json"),
            "before":obj["baseline"]["metrics"]["overall"],
            "guarded":obj["guarded"]["metrics"]["overall"]},ensure_ascii=False))
        return obj
    world=sim.register(target)
    initial=target/"initial.sqlite3"
    if not initial.is_file():
        world.snapshot(initial)
        with world._connection(initial) as c:
            c.execute("delete from world_events where profile<>?",(sim.WHO,))
    sha=hashlib.sha256(initial.read_bytes()).hexdigest()
    os.environ["FOURTH_SANDBOX_BASE_DB"]=str(initial)
    os.environ["FOURTH_DRIFT_DAY"]=scenario
    os.environ["FOURTH_COGNITIVE_LOOP"]="1"
    os.environ["FOURTH_SELF_REVISION"]="1"
    os.environ["FOURTH_TASK_DECAY_DAYS"]="0"
    state={}
    started=time.monotonic()
    original_bias=revision.bias
    start=1792040000.
    for name in ("baseline","guarded"):
        folder=target/name
        folder.mkdir(exist_ok=True)
        stored=target/(name+".json")
        if stored.is_file():
            arm=json.loads(stored.read_text(encoding="utf8"))
            state[name]=arm
            continue
        counts={"suspended":0,"contradicted":0,"stale_evidence":0,
                "maximum_age":0,"active_with_approved_bias":0}
        def guarded_bias(c,profile,context,at,prob):
            result=guard.guarded_bias(original_bias,c,profile,context,at,prob)
            if result.get("suppression"):
                status=result["suppression"]["status"]
                counts["suspended"]+=1
                counts[status]+=1
            elif result.get("changed"):
                counts["active_with_approved_bias"]+=1
            return result
        revision.bias=guarded_bias if name=="guarded" else original_bias
        try:
            arm=sim.run_arm(folder,world,start,seed,True,2)
        finally:
            revision.bias=original_bias
        with world._connection(Path(arm["db"])) as c:
            count=c.execute("""select count(*) from world_events
                                where profile in ('iincho_n','iincho','public')""").fetchone()[0]
            assert count==0
        assert arm["actions"]==18*24
        arm["guard_counts"]=counts
        arm["metrics"]=assess_trace(arm["trace"],shift_day)
        stored.write_text(json.dumps(arm,ensure_ascii=False),encoding="utf8")
        state[name]=arm
        print("ARM",seed,scenario,name,arm["metrics"]["overall"],
              "suspended",counts["suspended"],flush=True)
    b,g=state["baseline"],state["guarded"]
    diff=sum(x["chosen"]!=y["chosen"] for x,y in zip(b["trace"],g["trace"]))
    result={"schema":"fourth_guard_online_A_B_v1","seed":seed,"scenario":scenario,
        "shift_day":shift_day,"days":sim.DAYS,"paired_start_sha":sha,
        "all_results_synthetic":True,"N_persona_not_changed":True,
        "same_hidden_draws_by_action_and_step":True,
        "no_unselected_rewards_as_evidence":True,"llm_api_calls":0,
        "different_actions":diff,
        "baseline":{"metrics":b["metrics"],"strategy_reviews":b["strategy_review_log"]},
        "guarded":{"metrics":g["metrics"],"strategy_reviews":g["strategy_review_log"],
                   "safety":g["guard_counts"]},
        "duration_s":round(time.monotonic()-started,2)}
    (target/"comparison.json").write_text(
       json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"DONE":str(target/"comparison.json"),"seed":seed,"case":scenario,
       "old":b["metrics"]["overall"],"guarded":g["metrics"]["overall"],
       "guard":g["guard_counts"],"changed":diff},ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--scenario",choices=["none","7","10","14"],required=True)
    args=p.parse_args();run(args.seed,args.scenario)