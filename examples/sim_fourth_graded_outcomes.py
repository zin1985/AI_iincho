"""Fast-forward A/B for fourth test persona with GRADED task outcomes.

9 days, three rotating contexts; only chosen actions are rewarded.
The task environment uses paired per-step/action noise for fair comparisons.
World application and 9+3 evidence run through existing daino code. Entirely
isolated SQLite paths and ephemeral persona registry; no N memory or API.
"""
from __future__ import annotations
import json,os,random,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WHO="sim_fourth"
ACTIONS=("rest","wander","play")
DAY_BLOCK=3
DAYS=int(os.environ.get("FOURTH_LONG_DAYS","60"))
STAGES=(
  {"key":"material_observation","title":"水辺と素材を観察する",
   "teacher":"wander","quality":{"rest":.18,"wander":.38,"play":.82}},
  {"key":"playful_experiment","title":"遊びながら方法を見つける",
   "teacher":"play","quality":{"rest":.16,"wander":.83,"play":.32}},
  {"key":"recovery_and_revision","title":"休息して試行を組み直す",
   "teacher":"rest","quality":{"rest":.81,"wander":.36,"play":.2}},
)
# The environment's outcome probabilities are not visible to the policy.
# 0.22 random action exploration is identical in both A/B arms.
EPSILON=.22
SOURCE="simulation_environment"

def register(folder):
    folder.mkdir(parents=True,exist_ok=True)
    config=folder/"registry.json"
    config.write_text(json.dumps({"personas":[{"id":WHO,
        "name":"実験4号（隔離）","initial_place":"garden","initial_action":"観察する",
        "world_weights":[2,3,3,2,3,3,2,3]}]},ensure_ascii=False),
        encoding="utf8")
    os.environ["DAINO_PERSONA_REGISTRY"]=str(config)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    import persona_registry,world_space
    assert WHO in world_space.PROFILES
    return world_space

def _sample_demo(learning,n=2):
    samples=[]
    for index,phase in enumerate(STAGES):
        feat=(1.,0.,0.,0.,0.,0.,0.,.2,0.,0.,.2)+tuple(
            learning._hashed([phase["title"]]))+tuple(
            learning._hashed(["観察と試作の判断"] ))
        for j in range(n):
            samples.append((index*100+j,feat,ACTIONS,phase["teacher"]))
    return learning.train(samples)

def _model_distribution(learning,weights,features,world_c,now):
    base=learning._softmax(weights,features,ACTIONS)
    # Existing outcome head: world operation applied != task success.
    # Still used as an authentic part of the production baseline.
    raw={}
    for action in ACTIONS:
        prior,*_=learning._verified_action_value(world_c,WHO,now,action,None)
        raw[action]=base[action]*(0.65+.7*prior)
    s=sum(raw.values())
    return {a:raw[a]/s for a in ACTIONS}

def _mix_exploration(p):
    return {a:(1-EPSILON)*p[a]+EPSILON/len(ACTIONS) for a in ACTIONS}

def _roulette(p,roll):
    count=0.
    for action in ACTIONS:
        count+=p[action]
        if roll<count:return action
    return ACTIONS[-1]

def _set_goal(c,goal,at,idx):
    import self_growth
    self_growth.ensure(c)
    current=c.execute("""select title from self_growth_goals
        where profile=? and status='active' limit 1""",(WHO,)).fetchone()
    if current and current["title"]==goal:return
    c.execute("""update self_growth_goals set status='closed',updated_at=?
                 where profile=? and status='active'""",(at,WHO))
    c.execute("""insert into self_growth_goals
        (profile,title,goal_key,status,confidence,evidence_count,
        created_at,updated_at,origin_kind,origin_id,project_id,project_type)
        values(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (WHO,goal,"experiment_goal_"+str(idx)+"_"+str(int(at)),"active",85,1,at,at,
         "synthetic_test_context_not_self_generated",idx+1,None,None))

def run_arm(folder,world,start,seed,integrated,demonstrations=2):
    import world_growth
    import small_brain_goal_learning as learn
    import small_brain_policy as policy
    import brain_9plus3_policy_bridge as nine
    import brain_9plus3_micro_action_feedback as micro
    import brain_9plus3_task_learning as tasks
    import self_growth
    import brain_9plus3_cognitive_loop as cognitive
    decay_days=float(os.environ.get("FOURTH_TASK_DECAY_DAYS","0"))
    active_cognition=integrated and os.environ.get("FOURTH_COGNITIVE_LOOP","0")=="1"
    path=folder/("with_9plus3.sqlite3" if integrated else "without_9plus3.sqlite3")
    world.snapshot(path)
    with world._connection(path) as c:
        policy.ensure(c);learn.ensure(c);micro.ensure(c);tasks.ensure(c)
        self_growth.ensure(c)
    weights=_sample_demo(learn,demonstrations)
    randomizer=random.Random(seed)
    rolls=[randomizer.random() for _ in range(DAYS*24)]
    # Oracle noise keyed by step/action, identical across arms, but unchosen
    # outcomes are NEVER stored nor used in inference.
    outcomes=[{a:randomizer.random() for a in ACTIONS} for _ in rolls]
    trace=[]; checkpoints=[];inner_observation_log=[]
    checked=0;material_changes=0;previous_signature=None
    for slot in range(DAYS*144):
        now=start+slot*600.
        day=slot//144
        # The isolated world changes only on hour boundaries; the other
        # five ten-minute slots use the still-valid cached signature,
        # removing thousands of redundant SQLite connections.
        if slot%6:
            checked+=1
            continue
        with world._connection(path) as observer:
            latest=observer.execute("""select max(cycle_id)
                 from small_brain_goal_experiences where profile=?""",
                 (WHO,)).fetchone()[0]
            observed_goal=observer.execute("""select title from self_growth_goals
                 where profile=? and status='active' order by id desc limit 1""",
                 (WHO,)).fetchone()
            signature=(day//DAY_BLOCK,latest,
                       observed_goal["title"] if observed_goal else None)
            checked+=1
            if signature!=previous_signature:material_changes+=1
            previous_signature=signature
        if slot%6:continue
        index=slot//6
        stage_idx=(day//DAY_BLOCK)%len(STAGES)
        stage=STAGES[stage_idx]
        task_id="scenario:"+stage["key"]
        # After the midpoint, external conditions change: what used to
        # solve a problem is no longer best. The policy sees only actual
        # attempted outcomes, never the hidden environmental change.
        reversed_conditions=day>=DAYS//2
        quality=({a:stage["quality"][a] for a in ACTIONS} if not reversed_conditions
                 else {a:stage["quality"][ACTIONS[(ACTIONS.index(a)+1)%3]]
                       for a in ACTIONS})
        with world._connection(path) as c:
            _set_goal(c,stage["title"],now,stage_idx)
            current=learn.snapshot(c,WHO,now,(1.,0.,0.,0.,0.),None)
            base=_model_distribution(learn,weights,current["features"],c,now)
            reflection=(cognitive.inspect(c,WHO,now,task_id,
                                          stage["title"],base)
                        if active_cognition else None)
            adaptive_decay=(3. if reflection and
                    reflection["three_integrators"]["world_causal_model"]["possible_environment_change"]
                    else decay_days)
            raw=nine.counterfactual(c,WHO,now,None,base,
                                    task_context=task_id,
                                    task_recency_half_life_days=adaptive_decay)
            choice=raw["integrated"] if integrated else base
            current_epsilon=(reflection["exploration_epsilon"]
                             if reflection else EPSILON)
            probabilities={a:(1-current_epsilon)*choice[a]+
                            current_epsilon/len(ACTIONS) for a in ACTIONS}
            if reflection and (index%24==0 or index in (719,720,721,1439)):
                inner_observation_log.append({"day":day,
                  "source_kind":"algorithmic_observation_not_personal_introspection",
                  "brain_evidence":reflection,
                  "selection_distribution":{k:round(v,4)
                       for k,v in probabilities.items()}})
            selected=_roulette(probabilities,rolls[index])
            belief=tasks.posterior(tasks.evidence(c,WHO,now,task_id),ACTIONS)
            prediction=belief["expected_task_success"][selected]
            ins=c.execute("""insert into small_brain_cycles(
                profile,ts,outcome,action,project_id,probability,examples,
                evidence_json,features_json) values(?,?,?,?,?,?,?,?,?)""",
                (WHO,now,"reserved",selected,None,probabilities[selected],
                 demonstrations*3,"[]",json.dumps(current["features"])))
            cycle=int(ins.lastrowid)
        # Independent actual action validation; applied alone is NOT reward.
        reply=world_growth.apply(path,WHO,{
            "action":selected,"name":{"rest":"休息","wander":"探索","play":"遊び"}[selected],
            "reason":"隔離された高速学習実験の選択"},at=now)
        applied=bool(reply["ok"])
        passed=outcomes[index][selected]<quality[selected]
        with world._connection(path) as c:
            learn.observe(c,cycle,WHO,now,selected,
                "applied" if applied else "rejected",
                {"snapshot":current,"examples":[]},None,None)
            c.execute("update small_brain_cycles set outcome=? where id=?",
                      ("applied" if applied else "rejected",cycle))
            event=c.execute("""select id from world_events
                where profile=? and ts=? order by id desc limit 1""",
                (WHO,now)).fetchone()
            assert event and applied
            tasks.record(c,WHO,now,task_id,selected,int(event["id"]),passed,
                         source_kind=SOURCE,evaluator_id="hidden_task_quality",
                         predicted_prior=prediction)
        trace.append({"step":index,"day":day,"phase":stage_idx,
                     "chosen":selected,"passed":passed,"model_prior":round(prediction,4),
                     "preferred":stage["teacher"],
                     "optimal":max(quality,key=quality.get),
                     "regime":"changed" if reversed_conditions else "original"})
        if (index%72 in (0,12,24,48,71) or
            index in (0,DAYS*24//2,DAYS*24//2+24,DAYS*24-1)):
            checkpoints.append({
                "step":index,"day":round(index/24.,2),"goal":stage["title"],
                "without_9plus3":{a:round(v,4) for a,v in base.items()},
                "with_9plus3":{a:round(v,4) for a,v in raw["integrated"].items()},
                "task_evidence":raw["evidence"].get("objective_task",{}).get("observed",0),
                "selected":selected,
                "sources_count":len(raw["evidence"]["sources"])})
    blocks=[]
    for idx,stage in enumerate(STAGES):
        r=[x for x in trace if x["phase"]==idx]
        early=r[:24];late=r[-24:]
        def window(items):
            if not items:return {"success_rate":None,"optimal_action_rate":None,
                                 "goal_action_rate":None}
            return {"success_rate":round(sum(x["passed"] for x in items)/len(items),4),
                    "optimal_action_rate":round(sum(x["chosen"]==x["optimal"] for x in items)/len(items),4),
                    "goal_action_rate":round(sum(x["chosen"]==x["preferred"] for x in items)/len(items),4)}
        blocks.append({"goal":stage["title"],"initial_oracle":stage["quality"],
            "early":window(early),"late":window(late),
            "original_regime":window([x for x in r if x["regime"]=="original"]),
            "changed_regime":window([x for x in r if x["regime"]=="changed"]),
            "all":window(r),
            "action_counts":{a:sum(x["chosen"]==a for x in r) for a in ACTIONS}})
    quarters=[]
    for k in range(4):
        r=[x for x in trace if (k*DAYS/4)<=x["day"]<((k+1)*DAYS/4)]
        quarters.append({"quarter":k+1,"success_rate":round(sum(x["passed"] for x in r)/len(r),4),
            "optimal_action_rate":round(sum(x["chosen"]==x["optimal"] for x in r)/len(r),4),
            "n":len(r)})
    with world._connection(path) as c:
        objective_count=c.execute("""select count(*) from brain_9plus3_task_outcomes
                   where profile=?""",(WHO,)).fetchone()[0]
        micro_count=micro.status(c,WHO,limit=1)["count"]
        outsiders=c.execute("""select count(*) from world_events
                      where profile in ('iincho','iincho_n','public')""").fetchone()[0]
    return {"arm":"9plus3_with_verified_task_outcomes" if integrated else "control_without_9plus3",
        "db":str(path),"virtual_days":DAYS,
        "checks_every_ten_minutes":checked,
        "meaningful_change_checks":material_changes,
        "quiet_checks":checked-material_changes,
        "actions":len(trace),"objective_task_outcomes":objective_count,
        "micro_nine_three_feedback":micro_count,
        "unrelated_persona_events":outsiders,
        "success_rate":round(sum(x["passed"] for x in trace)/len(trace),4),
        "optimal_rate":round(sum(x["chosen"]==x["optimal"] for x in trace)/len(trace),4),
        "per_phase":blocks,"quarters":quarters,"trace":trace,
        "cognitive_9plus3_diary":inner_observation_log,
        "active_cognition_enabled":active_cognition,
        "checkpoints":checkpoints}

def run(seed=701,demonstrations=2,output=None):
    target=Path(output or ROOT/"reports"/"simulations"/
                ("fourth_outcomes_"+str(int(time.time()))+"_"+str(seed)))
    world=register(target)
    base_time=1791940000.+100000
    control=run_arm(target,world,base_time,seed,False,demonstrations)
    integrated=run_arm(target,world,base_time,seed,True,demonstrations)
    changes=[i for i,(a,b) in enumerate(zip(control["trace"],integrated["trace"]))
             if a["chosen"]!=b["chosen"]]
    result={"scenario":"fourth_persona_graded_outcomes","seed":seed,
            "registered_only_in_sandbox":True,
            "synthetic_training_demonstrations_per_goal":demonstrations,
            "synthetic_task_oracle_known_only_to_evaluator":True,
            "counterfactual_outcomes_not_exposed_to_learning":True,
            "independent_world_event_confirmation":True,
            "virtual_days":DAYS,"check_interval_minutes":10,
            "task_decay_days":float(os.environ.get("FOURTH_TASK_DECAY_DAYS","0")),
            "active_cognition":os.environ.get("FOURTH_COGNITIVE_LOOP","0")=="1",
            "control":control,"integrated":integrated,
            "changed_action_count":len(changes),
            "first_differences":changes[:15],
            "success_rate_delta":round(integrated["success_rate"]-control["success_rate"],4),
            "optimal_action_rate_delta":round(integrated["optimal_rate"]-control["optimal_rate"],4),
            "external_api_calls":0}
    (target/"report.json").write_text(json.dumps(result,ensure_ascii=False,
                                     indent=2),encoding="utf8")
    print(json.dumps({"report":str(target/"report.json"),"seed":seed,
      "virtual_days":DAYS,"ticks":DAYS*144,
      "quarters_control":control["quarters"],"quarters_integrated":integrated["quarters"],
      "actions_per_arm":DAYS*24,"changed_actions":len(changes),
      "success_rate_control":control["success_rate"],
      "success_rate_integrated":integrated["success_rate"],
      "optimal_rate_control":control["optimal_rate"],
      "optimal_rate_integrated":integrated["optimal_rate"],
      "per_phase":[{"goal":x["goal"],
          "early":x["early"],"late":x["late"],
          "delta_success":round(y["all"]["success_rate"]-x["all"]["success_rate"],4)}
           for x,y in zip(control["per_phase"],integrated["per_phase"])],
      "objective_task_outcomes_per_arm":integrated["objective_task_outcomes"],
      "micro_nine_three_feedback_per_arm":integrated["micro_nine_three_feedback"],
      "external_api_calls":0},ensure_ascii=False))
    return result

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=701)
    parser.add_argument("--demos",type=int,default=2)
    opts=parser.parse_args()
    run(seed=opts.seed,demonstrations=opts.demos)
