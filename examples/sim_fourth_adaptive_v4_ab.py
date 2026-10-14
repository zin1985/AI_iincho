"""Measured fourth-persona A/B: safety guard only vs guard+adaptive learning.

Identical initial DB and RNG, 18 virtual days, actual chosen outcomes only.
Never uses an LLM, Ollama, N persona records or production SQLite.
"""
from __future__ import annotations
import argparse,os,hashlib,json,time
from pathlib import Path
import brain_9plus3_self_revision as revision
import brain_9plus3_strategy_guard_v2 as guard
from tools import sim_fourth_graded_outcomes as sim
from tools.sim_fourth_guard_v2_ab import assess_trace

ROOT=Path(__file__).resolve().parents[1]
def run(seed,scenario):
    assert scenario in ("none","7","10","14")
    assert sim.DAYS==18
    shift=None if scenario=="none" else int(scenario)
    folder=ROOT/"reports"/"simulations"/f"fourth_adaptive_v4_{scenario}_{seed}"
    final=folder/"comparison.json"
    if final.is_file():
        report=json.loads(final.read_text(encoding="utf8"))
        print("REUSE",final,flush=True)
        return report
    world=sim.register(folder)
    source=folder/"initial.sqlite3"
    if not source.exists():
        world.snapshot(source)
        with world._connection(source) as c:
            c.execute("delete from world_events where profile<>?",(sim.WHO,))
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    os.environ.update({"FOURTH_SANDBOX_BASE_DB":str(source),
       "FOURTH_DRIFT_DAY":scenario,"FOURTH_COGNITIVE_LOOP":"1",
       "FOURTH_SELF_REVISION":"1","FOURTH_TASK_DECAY_DAYS":"0"})
    original=revision.bias
    result={}
    for mode in ("guard_only","guard_adaptive"):
        f=folder/(mode+".json")
        if f.exists():
            result[mode]=json.loads(f.read_text(encoding="utf8"))
            continue
        os.environ["FOURTH_ADAPTIVE_REGIME"]="1" if mode=="guard_adaptive" else "0"
        counts={"suppressed":0,"active":0}
        def wrapped(c,profile,context,at,prob):
            value=guard.guarded_bias(original,c,profile,context,at,prob)
            if value.get("suppression"):counts["suppressed"]+=1
            if value.get("changed"):counts["active"]+=1
            return value
        revision.bias=wrapped
        try:
            arm=sim.run_arm(folder/mode,world,1792040000.,seed,True,2)
        finally:
            revision.bias=original
        assert arm["unrelated_persona_events"]==0
        assert arm["actions"]==432 and arm["objective_task_outcomes"]==432
        arm["metrics"]=assess_trace(arm["trace"],shift)
        arm["guard_counts"]=counts
        f.write_text(json.dumps(arm,ensure_ascii=False),encoding="utf8")
        result[mode]=arm
        print("ARM",seed,scenario,mode,arm["metrics"]["overall"],
             "regime_events",len(arm["regime_log"]),flush=True)
    old,new=result["guard_only"],result["guard_adaptive"]
    changed=sum(a["chosen"]!=b["chosen"] for a,b in zip(old["trace"],new["trace"]))
    report={"study":"fourth_adaptive_controller_v4","scenario":scenario,
       "seed":seed,"virtual_days":18,"initial_sha256":sha,
       "identical_oracle_draws":True,"sandbox_only":True,"llm_api_calls":0,
       "new_outcomes_not_shared_across_arms":True,
       "different_actions":changed,
       "guard_only":{"metrics":old["metrics"],"guard_counts":old["guard_counts"]},
       "guard_adaptive":{"metrics":new["metrics"],"guard_counts":new["guard_counts"],
                         "regime_alerts":new["regime_log"]}}
    final.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("COMPLETE",final,"changed",changed,flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--scenario",choices=["none","7","10","14"],required=True)
    opts=p.parse_args()
    run(opts.seed,opts.scenario)