"""Independent checks of simulated fourth persona utterance provenance.

The text is a deterministic probe, NOT the real N/fourth LLM, and these tests
must never write the production DB or call a language API.
"""
from pathlib import Path
import sqlite3
import pytest
import brain_9plus3_lived_learning as lived
from tools import sim_fourth_speech_reflection_60d as sim

def test_all_paired_dbs_durable_and_identity_isolated():
    for seed in (991,992,993):
        for scene in ("stationary","shift30"):
            for arm in ("baseline","extended"):
                file=sim.SOURCE/f"{seed}_{scene}_{arm}.sqlite3"
                with sqlite3.connect("file:"+file.as_posix()+"?mode=ro",uri=True) as c:
                    total=c.execute("select count(*) from world_events").fetchone()[0]
                    learned=c.execute("select count(*) from brain_9plus3_lived_events").fetchone()[0]
                    assert total>=972
                    assert learned>0
                    assert c.execute("""select count(*) from world_events
                          where profile!='sim_fourth'""").fetchone()[0]==0
                    assert c.execute("""select count(*) from brain_9plus3_lived_events
                          where task_success is not null""").fetchone()[0]==0

def test_speech_reflection_have_grounded_source_and_no_future_evidence():
    for seed in (991,992,993):
        for scene in ("stationary","shift30"):
            for arm in ("baseline","extended"):
                result=sim.experiment(seed,scene,arm)
                assert len(result["days"])==60
                assert result["metrics"]["spoken_days"]==60
                for day in (0,5,29,30,39,59):
                    row=result["days"][day]
                    assert row["sources"]
                    assert row["source_action"] in row["speech"]
                    assert row["sources"][0] in row["reflection"]
                    assert "結果の良し悪しは、まだ分からない" in row["speech"]
                    assert "次に何を試すべきかは、この記録だけでは決められない" in row["reflection"]
                    with sqlite3.connect("file:"+Path(result["db"]).as_posix()+"?mode=ro",uri=True) as c:
                        c.row_factory=sqlite3.Row
                        sources=sim._events(c,sim.T0+(day+1)*sim.DAY-1)
                        assert all(ev["ts"]<=sim.T0+(day+1)*sim.DAY-1 for ev in sources)
                        assert row["sources"][0] in {ev["source_id"] for ev in sources}
                    assert row["utterance_type"]=="rule_surrogate_not_LLM"
                    assert row["independent_grade_count"]==0

def test_added_facets_enable_craft_focus_without_changing_personal_goals():
    old=sim.experiment(992,"shift30","baseline")
    new=sim.experiment(992,"shift30","extended")
    assert old["metrics"]["craft_days_last30"]==0
    assert new["metrics"]["craft_days_last30"]>=20
    assert old["metrics"]["distinct_topics"]<=3
    assert new["metrics"]["distinct_topics"]>=6
    assert old["metrics"]["unapproved_goal_changes"]==0
    assert new["metrics"]["unapproved_goal_changes"]==0

def test_integrator_provenance_poisoning_is_rejected():
    row={
      "source_profile":"sim_fourth","source_id":"world_events#1",
      "source_event":1,"task_success":None,
      "source_action":"ゲームで遊ぶ","action":"craft","context_key":"activity:craft",
      "region_json":"{}","integrators_json":"{}"}
    with pytest.raises((KeyError,AssertionError)):
        sim._verify(row)

def test_missing_evidence_can_mean_silence():
    import sqlite3
    with sqlite3.connect(":memory:") as c:
        c.row_factory=sqlite3.Row
        c.executescript("""create table world_events(id integer primary key,profile text,ts real,action text);
        create table brain_9plus3_lived_events(id integer primary key,
          profile text,source_id text,ts real,context_key text,action text,
          task_success integer,region_json text,integrators_json text);""")
        r=sim._candidate(0,c)
        assert r["speak"] is False and r["speech"]==""
