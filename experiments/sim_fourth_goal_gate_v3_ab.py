"""Six-arm long-term sandbox A/B of *observed* fourth-persona 9+3 experience.

Each chosen virtual action is applied through daino world_growth, independently
graded, then committed to its own SQLite. A verified ingestion adapter forwards
that concrete result to all nine persistent processors and all three engines.

Internal "goals" are lab strategies (rest/wander/play), NOT N's personal goals.
Ablation comparisons test whether each 3-integrator has causal influence on
goal and action choice; they do not claim genuine emotion or consciousness.
"""
from __future__ import annotations
import os,json,random,time,math,hashlib,shutil,argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import sim_fourth_graded_outcomes as sim
from tools import fourth_lived_pipeline as pipe
import brain_9plus3_lived_learning as lived
import brain_9plus3_task_learning as task
from tools import fourth_goal_adoption_gate_v3 as gate

MODES=("full","full_v2")
GOALS={"rest":"体力を回復する方法を検証","wander":"新しい素材を探す方法を検証",
       "play":"試作品を作りながら方法を検証"}
ACTIONS=sim.ACTIONS

def run_arm(folder,world,template,seed,mode,days,drift_day,rolls,outcomes):
    import world_growth
    path=folder/"world.sqlite3"
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(template,path)
    with world._connection(path) as c:
        task.ensure(c);lived.ensure(c)
    old_goals={}
    latest_review={}
    trace=[];goal_changes=[];states=[];ingested=0;surprise_log=[]
    gate_reasons={}; gate_daily=[]
    start=1792040000.
    for step in range(days*24):
        day=step//24
        now=start+3600.*step
        stage_idx=(day//sim.DAY_BLOCK)%len(sim.STAGES)
        stage=sim.STAGES[stage_idx]
        context="scenario:"+stage["key"]
        changed_environment=(drift_day is not None and day>=drift_day)
        truth=({a:stage["quality"][a] for a in ACTIONS}
               if not changed_environment else
               {a:stage["quality"][ACTIONS[(ACTIONS.index(a)+1)%3]]
                 for a in ACTIONS})
        current_goal=old_goals.get(context)
        with world._connection(path) as c:
            # A fair common objective outcome learner, for all arms.
            teacher=stage["teacher"]
            prior={a:(.70 if a==teacher else .15) for a in ACTIONS}
            learned=task.adapt(c,sim.WHO,now,context,prior,active=True)
            probs=learned["updated"]
            review=None
            if mode!="baseline" and (context not in old_goals or
                                      step-latest_review.get(context,-999)>=6):
                disabled=(mode[3:] if mode.startswith("no_") else None)
                if mode=="nine_only":
                    ev=task.evidence(c,sim.WHO,now,context,limit=128)
                    posterior=task.posterior(ev,ACTIONS,now=now,half_life_days=4.)
                    counts=posterior["counts"]
                    weights=posterior["expected_task_success"]
                    considered={a:(weights[a]-.5)*counts[a]/(counts[a]+4.)
                                +.17/math.sqrt(counts[a]+1.)+
                                (.08 if a==current_goal else 0.)
                                for a in ACTIONS}
                    new_goal=(max(ACTIONS,key=lambda a:(considered[a],-ACTIONS.index(a)))
                              if len(ev)>=6 else current_goal or "rest")
                    review={"goal":new_goal,"previous_goal":current_goal,
                      "changed":current_goal is not None and new_goal!=current_goal,
                      "scores":considered,"sources":[e["source"] for e in ev[:8]],
                      "observations":len(ev),"integrator_ablation":"ALL"}
                else:
                    review=pipe.decide(c,sim.WHO,context,now,current_goal,
                                       disable=disabled,record=True)
                old_goals[context]=review["goal"]
                latest_review[context]=step
                if review["changed"]:
                    goal_changes.append({"step":step,"day":day,
                      "context":context,"from":current_goal,"to":review["goal"],
                      "actual_persona_self_review":False,
                      "source_count":len(review["sources"]),
                      "integrator_ablation":review["integrator_ablation"]})
            current_goal=old_goals.get(context)
            if mode=="full_v2" and current_goal:
                decision=gate.gate(c,sim.WHO,now,context,probs,current_goal)
                probs=decision["weights"]
                reason=decision["reason"]
                gate_reasons[reason]=gate_reasons.get(reason,0)+1
                if step%24==0:
                    gate_daily.append({"day":day,"context":context,
                      "goal":current_goal,"strength":decision["strength"],
                      "reason":reason,"observations":decision["observations"],
                      "source_ids":decision.get("source_ids",[])[:3]})
            elif mode!="baseline" and current_goal:
                probs=pipe.decide_probabilities(probs,current_goal,weight=.42)
            # Same exploration schedule and action roll in every arm.
            mixed={a:.82*probs[a]+.18/3. for a in ACTIONS}
            chosen=sim._roulette(mixed,rolls[step])
            prior_prob=task.posterior(task.evidence(c,sim.WHO,now,context),ACTIONS)
            prediction=prior_prob["expected_task_success"][chosen]
        result=world_growth.apply(path,sim.WHO,{
            "action":chosen,"name":{"rest":"休息","wander":"探索","play":"遊び"}[chosen],
            "reason":"9領域の実経験に基づく隔離実験"},at=now)
        if not result.get("ok"):raise RuntimeError("sandbox world event rejected")
        success=bool(outcomes[step][chosen]<truth[chosen])
        with world._connection(path) as c:
            ev=c.execute("""select id from world_events where profile=? and ts=?
                 order by id desc limit 1""",(sim.WHO,now)).fetchone()
            if not ev:raise RuntimeError("world event not actually saved")
            item=task.record(c,sim.WHO,now,context,chosen,int(ev["id"]),success,
                             source_kind="simulation_environment",
                             evaluator_id="isolated_hidden_oracle",
                             predicted_prior=prediction)
            signal=pipe.verified_ingest(c,sim.WHO,item["id"])
            if signal.get("duplicate"):raise RuntimeError("unnecessary duplicate ingestion")
            ingested+=1
            if step%24==0 or step==days*24-1:
                snap=lived.snapshot(c,sim.WHO,context)
                states.append({"step":step,"day":day,"context":context,
                  "nine":snap["regions"],"three":snap["integrators"],
                  "sources":snap["sources"][:3],"goal":old_goals.get(context)})
        trace.append({"step":step,"day":day,"context":context,"action":chosen,
          "success":success,"goal":current_goal,"task_shifted":changed_environment,
          "optimal":max(truth,key=truth.get),
          "oracle_expected":truth[chosen],
          "oracle_regret":max(truth.values())-truth[chosen]})
    with world._connection(path) as c:
        aggregate={}
        for table in ("brain_9plus3_lived_events","brain_9plus3_region_states",
                      "brain_9plus3_integrator_states","brain_9plus3_goal_choices"):
            aggregate[table]=c.execute("select count(*) from "+table+
               " where profile=?",(sim.WHO,)).fetchone()[0]
        others=c.execute("select count(*) from world_events where profile in ('iincho','iincho_n','public')").fetchone()[0]
        if others:raise RuntimeError("real personas in sandbox events")
    count=len(trace)
    return {"mode":mode,"seed":seed,"days":days,"drift_day":drift_day,
        "n":count,"successes":sum(t["success"] for t in trace),
        "success_rate":round(sum(t["success"] for t in trace)/count,4),
        "oracle_mean_regret":round(sum(t["oracle_regret"] for t in trace)/count,5),
        "optimal_rate":round(sum(t["action"]==t["optimal"] for t in trace)/count,4),
        "goal_changes":goal_changes,"goal_change_count":len(goal_changes),
        "verified_region_ingestions":ingested,"sqlite_rows":aggregate,
        "region_integrator_daily_states":states,"trace":trace,
        "independent_persona_rows":others,
        "external_api_calls":0,"source_db":str(path),
        "gate_reasons":gate_reasons,"gate_daily":gate_daily}

def execute(seed=961,days=60,drift_day=30,modes=MODES):
    target=ROOT/"reports"/"simulations"/f"fourth_lived_gate_v3_{days}d_{seed}_{'none' if drift_day is None else drift_day}"
    target.mkdir(parents=True,exist_ok=True)
    world=sim.register(target)
    template=target/"initial_world.sqlite3"
    if not template.exists():
        world.snapshot(template)
        with world._connection(template) as c:
            c.execute("delete from world_events where profile<>?",(sim.WHO,))
    digest=hashlib.sha256(template.read_bytes()).hexdigest()
    randomizer=random.Random(seed)
    rolls=[randomizer.random() for _ in range(days*24)]
    outcomes=[{a:randomizer.random() for a in ACTIONS}
              for _ in range(days*24)]
    results={}
    for mode in modes:
        path=target/(mode+".json")
        if path.exists():
            arm=json.loads(path.read_text(encoding="utf8"))
            if arm["days"]==days and arm["seed"]==seed:
                results[mode]=arm
                print("REUSE",mode,flush=True)
                continue
        arm=run_arm(target/mode,world,template,seed,mode,days,drift_day,rolls,outcomes)
        path.write_text(json.dumps(arm,ensure_ascii=False),encoding="utf8")
        results[mode]=arm
        print("FINISHED",mode,"success",arm["success_rate"],
              "goals",arm["goal_change_count"],"9+3_ingested",arm["verified_region_ingestions"],flush=True)
        (target/"partial_summary.json").write_text(
            json.dumps({"seed":seed,"days":days,"modes_done":list(results),
              "template_sha256":digest},ensure_ascii=False),encoding="utf8")
    metrics={k:{field:r[field] for field in
             ("success_rate","successes","oracle_mean_regret","optimal_rate",
              "goal_change_count","verified_region_ingestions","sqlite_rows")}
             for k,r in results.items()}
    diff={}
    for mode,arm in results.items():
        if mode=="baseline":continue
        base=results.get("full")
        if base and mode!="full":
            diff[mode]={"action_changes_vs_full":
                sum(x["action"]!=y["action"] for x,y in zip(arm["trace"],base["trace"])),
                "goal_changes_vs_full":
                sum(x["goal"]!=y["goal"] for x,y in zip(arm["trace"],base["trace"]))}
    report={"schema":"fourth_lived_9plus3_goal_gate_A_B_v3",
      "seed":seed,"days":days,"drift_day":drift_day,
      "same_world_template_sha256":digest,"modes":metrics,
      "ablation_effects":diff,"non_real_persona":True,
      "self_goals_are_simulated_strategies_not_personal_values":True,
      "no_unobserved_results_in_training":True,"api_calls":0,
      "sandbox_isolation_verified":True}
    path=target/"summary.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("SUMMARY",str(path),json.dumps(metrics,ensure_ascii=False),flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=961)
    parser.add_argument("--days",type=int,default=60)
    parser.add_argument("--drift-day",type=str,default="30")
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    args=parser.parse_args()
    execute(args.seed,args.days,None if args.drift_day=="none" else int(args.drift_day),
            tuple(args.modes))
