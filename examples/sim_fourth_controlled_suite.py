"""A controlled four-arm study of the isolated fourth persona.

Every arm starts from the exact same SQLite snapshot. Only one arm is run
at a time to avoid overloading the user's main PC. No Ollama or LLM calls.
"""
from __future__ import annotations
import os,sys,json,time,sqlite3,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import sim_fourth_graded_outcomes as sim

def digest(file):
    h=hashlib.sha256()
    with open(file,"rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def run(seed):
    root=ROOT/"reports"/"simulations"/f"fourth_controlled_{sim.DAYS}d_{seed}"
    if root.exists() and (root/"controlled_report.json").exists():
        raise ValueError("finished controlled trial already exists")
    root.mkdir(parents=True,exist_ok=True)
    world=sim.register(root)
    initial=root/"seed_snapshot.sqlite3"
    if not initial.exists():
        world.snapshot(initial)
        with world._connection(initial) as c:
            # World stage histories for the three real personas are irrelevant.
            # The copy still includes neutral world agents and geography,
            # but no seed history from existing persona names.
            c.execute("delete from world_events where profile<>?",(sim.WHO,))
    initial_hash=digest(initial)
    os.environ["FOURTH_SANDBOX_BASE_DB"]=str(initial)
    modes=(("no_9plus3",False,False,False),
           ("outcome_9plus3",True,False,False),
           ("active_9plus3",True,True,False),
           ("active_with_review",True,True,True))
    partial=root/"controlled_report.partial.json"
    if partial.exists():
        saved=json.loads(partial.read_text(encoding="utf8"))
        if saved["seed"]!=seed or saved["days"]!=sim.DAYS or saved["template_sha256"]!=initial_hash:
            raise ValueError("partial experiment snapshot mismatch")
        results=saved["modes"]
    else:
        results={}
    started=time.monotonic()
    start=1792040000.
    for name,integrated,cognitive,revisions in modes:
        if name in results:
            print("REUSED_COMPLETED_ARM",name,flush=True)
            continue
        dest=root/name
        dest.mkdir(exist_ok=True)
        os.environ["FOURTH_COGNITIVE_LOOP"]="1" if cognitive else "0"
        os.environ["FOURTH_SELF_REVISION"]="1" if revisions else "0"
        arm=sim.run_arm(dest,world,start,seed,integrated,2)
        # Only the simulator persona may generate new world events.
        with world._connection(Path(arm["db"])) as c:
            extra=c.execute("""select count(*) from world_events where profile
               in ('iincho','iincho_n','public')""").fetchone()[0]
            present=c.execute("""select 1 from sqlite_master where type='table'
                 and name='brain_9plus3_persona_strategies'""").fetchone()
            learned=(c.execute("""select count(*) from brain_9plus3_persona_strategies
                  where profile=?""",(sim.WHO,)).fetchone()[0] if present else 0)
        assert extra==0,("unexpected real persona events",name,extra)
        assert arm["actions"]==sim.DAYS*24
        assert arm["checks_every_ten_minutes"]==sim.DAYS*144
        # Multiple accepted reviews can update the same task-specific
        # strategy. Number of strategy rows need not equal review history.
        if revisions and arm["strategy_review_log"]:
            assert 0<learned<=len(arm["strategy_review_log"])
        results[name]={"data":arm,"starting_snapshot_sha256":initial_hash,
           "strategy_count":learned,"other_persona_events":extra,
           "api_calls":0}
        # Incremental report after each completed arm, so even an interrupted
        # multi-arm study retains correctly labeled partial results.
        state={"seed":seed,"days":sim.DAYS,
             "template_sha256":initial_hash,"complete":False,
             "modes":results}
        (root/"controlled_report.partial.json").write_text(
             json.dumps(state,ensure_ascii=False),encoding="utf8")
        print("ARM_FINISHED",seed,name,arm["success_rate"],
              "reviews",len(arm["strategy_review_log"]),flush=True)
    final={"study":"exact_initial_world_four_arms_v1",
           "seed":seed,"virtual_days":sim.DAYS,
           "template_sha256":initial_hash,"same_hidden_task_oracle":True,
           "same_stochastic_action_rolls":True,
           "simulated_reviews_are_not_N_choices":True,
           "unselected_oracle_outcomes_never_train":True,
           "each_arm_uses_original_production_policy_code":True,
           "no_other_persona_events":True,
           "external_api_calls":0,
           "duration_seconds":round(time.monotonic()-started,2),
           "arms":results}
    out=root/"controlled_report.json"
    out.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf8")
    for name,item in results.items():
        d=item["data"]
        print("RESULT",name,"success",d["success_rate"],
           "optimal",d["optimal_rate"],"adoptions",len(d["strategy_review_log"]))
    print("COMPLETE_REPORT",out,flush=True)
    return final

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    run(args.seed)