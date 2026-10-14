"""Nine independent source-backed state processors and three integrating outputs."""
import json
import pytest
import world_space as world
import world_growth
import brain_9plus3_lived_learning as lived
import brain_9plus3_task_learning as task
from tools import fourth_lived_pipeline as verified_lived
from test_brain_9plus3_task_learning import make_db

CONTEXT="scenario:nine_area_learning"

def _fixture(tmp_path):
    db,now=make_db(tmp_path)
    with world._connection(db) as c:
        task.ensure(c)
        lived.ensure(c)
    return db,now

def _record(db,now,action,success,source="simulation_environment"):
    with world._connection(db) as c:
        assert world_growth.apply(db,"iincho_n",
          {"action":action,"name":"test","reason":"sandbox"},at=now)["ok"]
        row=c.execute("""select id from world_events
           where profile='iincho_n' and ts=? order by id desc limit 1""",
           (now,)).fetchone()
        assert row
        # N is a fixture identity in this isolated test DB; it still may
        # never consume simulation_environment reward.
        result=task.record(c,"iincho_n",now,CONTEXT,action,int(row["id"]),
            success,source_kind="verified_task_evaluator",
            evaluator_id="unit_test_no_production",predicted_prior=.58)
        verified_lived.verified_ingest(c,"iincho_n",result["id"])
        return result

def test_all_nine_and_three_are_persistently_changed_and_linked(tmp_path):
    db,t0=_fixture(tmp_path)
    updates=[]
    for i in range(26):
        _record(db,t0+i*3600,["rest","wander","play"][i%3],i%4!=0)
        with world._connection(db) as c:
            info=lived.snapshot(c,"iincho_n",CONTEXT)
            updates.append(info)
    assert all(v["all_nine_defined"] for v in updates)
    with world._connection(db) as c:
        rows=c.execute("""select source_id,region_json,integrators_json,
                 links_json from brain_9plus3_lived_events
                 where profile='iincho_n' order by id""").fetchall()
        assert len(rows)==26
        assert len({row["source_id"] for row in rows})==26
        for row in rows:
            assert set(json.loads(row["region_json"]))==set(lived.REGIONS)
            assert set(json.loads(row["integrators_json"]))==set(lived.INTEGRATORS)
            assert len(json.loads(row["links_json"]))>=12
        assert c.execute("""select count(*) from brain_9plus3_region_states
                 where profile='iincho_n'""").fetchone()[0]==9
        assert c.execute("""select count(*) from brain_9plus3_integrator_states
                 where profile='iincho_n'""").fetchone()[0]==3
        assert any(updates[0]["regions"][r]!=updates[-1]["regions"][r]
                   for r in lived.REGIONS)
        assert any(updates[0]["integrators"][r]!=updates[-1]["integrators"][r]
                   for r in lived.INTEGRATORS)
        assert lived.snapshot(c,"iincho",CONTEXT)["sources"]==[]

def test_operational_is_not_task_grade(tmp_path):
    db,t0=_fixture(tmp_path)
    with world._connection(db) as c:
        rec=lived.observe_operational(c,"iincho_n","operation:rest","rest",
                       t0,"small_brain_goal_experiences#777",True)
        assert rec["task_grade_known"] is False
        row=c.execute("select task_success,operational_applied from brain_9plus3_lived_events").fetchone()
        assert row["task_success"] is None and row["operational_applied"]==1
        same=lived.observe_operational(c,"iincho_n","operation:rest","rest",
                       t0,"small_brain_goal_experiences#777",True)
        assert same["duplicate"]
        assert c.execute("select count(*) from brain_9plus3_lived_events").fetchone()[0]==1

def test_no_synthetic_outcome_for_N(tmp_path):
    db,t0=_fixture(tmp_path)
    with world._connection(db) as c:
        with pytest.raises(ValueError):
            lived.observe_task(c,"iincho_n",CONTEXT,"play",t0,
                   "source#777",True,source_kind="simulation_environment")

def test_source_ownership_and_duplicate_dedup(tmp_path):
    db,t0=_fixture(tmp_path)
    with world._connection(db) as c:
        lived.observe_operational(c,"iincho_n","operation:rest","rest",t0,
                                  "verified#1",True)
        with pytest.raises(ValueError):
            lived.observe_operational(c,"iincho","operation:rest","rest",t0,
                                      "verified#1",True)
