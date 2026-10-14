"""Isolated, factual 9+3 experience source and causal ablation tests."""
import json,sqlite3
import pytest
from tools import fourth_lived_pipeline as pipe
import brain_9plus3_task_learning as task
import brain_9plus3_lived_learning as lived

def database():
    c=sqlite3.connect(":memory:")
    c.row_factory=sqlite3.Row
    c.execute("""create table world_events(id integer primary key,
      ts real not null,profile text not null,action text not null)""")
    task.ensure(c);lived.ensure(c)
    return c

def put(c,n,action,success,profile="sim_fourth"):
    names={"rest":"休息","wander":"探索","play":"遊び"}
    ts=100000.+n*3600.
    evt=c.execute("insert into world_events(ts,profile,action) values(?,?,?)",
                  (ts,profile,names[action]))
    grade=task.record(c,profile,ts,"scenario:verification",action,
                      evt.lastrowid,bool(success),
                      source_kind="simulation_environment",
                      evaluator_id="independent_world_oracle",
                      predicted_prior=.55)
    return grade

def test_explicit_source_owner_action_and_deduplication():
    c=database()
    try:
        row=put(c,0,"play",True)
        first=pipe.verified_ingest(c,"sim_fourth",row["id"])
        assert first["task_grade_known"] is True
        assert first["region_states"].keys()==set(lived.REGIONS)
        assert first["integrators"].keys()==set(lived.INTEGRATORS)
        assert len(first["links"])>=12
        assert pipe.verified_ingest(c,"sim_fourth",row["id"])["duplicate"]
        with pytest.raises(ValueError):
            pipe.verified_ingest(c,"iincho_n",row["id"])
        c.execute("update world_events set action='探索' where id=1")
        with pytest.raises(ValueError,match="world action"):
            pipe.verified_ingest(c,"sim_fourth",row["id"])
    finally:c.close()

def test_task_grade_tampering_rejected_after_ingest():
    c=database()
    try:
        item=put(c,1,"rest",True)
        pipe.verified_ingest(c,"sim_fourth",item["id"])
        c.execute("update brain_9plus3_task_outcomes set task_success=0 where id=?",
                 (item["id"],))
        with pytest.raises(ValueError,match="modified"):
            pipe.verified_ingest(c,"sim_fourth",item["id"])
    finally:c.close()

def test_nine_processors_three_engines_and_self_selected_goal():
    c=database()
    try:
        for i in range(45):
            a=("rest","wander","play")[i%3]
            # Reward, novelty and habit intentionally vary over time.
            grade=put(c,i,a,a=="play" or i%11==0)
            pipe.verified_ingest(c,"sim_fourth",grade["id"])
        state=lived.snapshot(c,"sim_fourth","scenario:verification")
        assert state["all_nine_defined"]
        assert len(state["sources"])==12
        for k in lived.REGIONS:assert 0.<=state["regions"][k]<=1.
        for k in lived.INTEGRATORS:assert 0.<=state["integrators"][k]<=1.
        rows=c.execute("select region,events from brain_9plus3_region_states").fetchall()
        assert {r["region"]:r["events"] for r in rows}=={k:45 for k in lived.REGIONS}
        rows=c.execute("select integrator,events from brain_9plus3_integrator_states").fetchall()
        assert {r["integrator"]:r["events"] for r in rows}=={k:45 for k in lived.INTEGRATORS}
        now=100000.+46*3600.
        full=pipe.decide(c,"sim_fourth","scenario:verification",now,
                         old_goal="rest",disable=None)
        assert full["goal"]=="play"
        assert full["changed"]
        for dimension in lived.INTEGRATORS:
            minus=pipe.decide(c,"sim_fourth","scenario:verification",now,
                         old_goal="rest",disable=dimension,record=False)
            assert minus["scores"]!=full["scores"]
            assert minus["engine_inputs"][dimension]==0.
        stored=c.execute("select count(*) from brain_9plus3_goal_choices").fetchone()[0]
        assert stored==1
        after=pipe.decide_probabilities({"rest":.4,"wander":.35,"play":.25},
                    full["goal"])
        assert after["play"]>.25 and abs(sum(after.values())-1.)<1e-8
        assert not lived.snapshot(c,"iincho_n","scenario:verification")["sources"]
    finally:c.close()

def test_each_of_nine_region_memories_affects_three_engine_next_step():
    """Counterfactual perturbation of a stored region changes an engine signal."""
    original=database()
    try:
        for i in range(32):
            action=("rest","wander","play")[i%3]
            row=put(original,i,action,(i%7)<4)
            pipe.verified_ingest(original,"sim_fourth",row["id"])
        original.commit()  # sqlite backup must not run inside writer transaction
        at=100000.+33*3600.
        def simulate(perturb=None):
            copy=sqlite3.connect(":memory:")
            copy.row_factory=sqlite3.Row
            original.backup(copy)
            try:
                if perturb:
                    r=copy.execute("""select value from brain_9plus3_region_states
                        where profile='sim_fourth' and context_key='scenario:verification'
                        and region=?""",(perturb,)).fetchone()
                    old=float(r["value"])
                    new=0.0 if old>.5 else 1.0
                    copy.execute("""update brain_9plus3_region_states set value=?
                       where profile='sim_fourth' and context_key='scenario:verification'
                       and region=?""",(new,perturb))
                return lived.observe_operational(copy,"sim_fourth",
                    "scenario:verification","play",at,"unit#new_event",True)["integrators"]
            finally:
                copy.close()
        baseline=simulate()
        effects={}
        for region in lived.REGIONS:
            changed=simulate(region)
            effects[region]=max(abs(changed[k]-baseline[k]) for k in lived.INTEGRATORS)
        assert set(effects)==set(lived.REGIONS)
        assert all(delta>1e-7 for delta in effects.values()),effects
    finally: original.close()
