"""N's world-action facts flow to persistent 9+3 without altering personal goals."""
import json
import pytest
import world_space as world
import n_nine_three_lived_sync as live

def _setup(tmp_path):
    p=tmp_path/"nine_live.sqlite3"
    world.snapshot(p)
    with world._connection(p) as c:
        c.execute("delete from world_events")
        c.executemany("""insert into world_events(ts,profile,place,action,detail)
                         values(?,?,?,?,?)""",[
          (1000.,"iincho_n","garden","小道を散歩する","observed"),
          (1001.,"iincho_n","playground","ゲームで遊ぶ","observed"),
          (1002.,"iincho_n","pond","静かに休む","observed"),
          (1003.,"iincho_n","library","空想にふける","not eligible"),
          (1004.,"iincho","garden","小道を散歩する","other persona")])
    return p

def test_production_ingest_persists_all_9_plus_3_and_no_rewards(tmp_path):
    p=_setup(tmp_path)
    first=live.sync(p)
    assert first["observed"]==3 and first["skipped"]==1
    with world._connection(p) as c:
        names=c.execute("select name from sqlite_master where type='table'").fetchall()
        assert c.execute("""select count(*) from brain_9plus3_region_states
            where profile='iincho_n'""").fetchone()[0]==27
        assert c.execute("""select count(*) from brain_9plus3_integrator_states
            where profile='iincho_n'""").fetchone()[0]==9
        assert c.execute("""select count(*) from brain_9plus3_lived_events
            where profile='iincho_n' and task_success is not null""").fetchone()[0]==0
        assert c.execute("""select count(*) from brain_9plus3_lived_events
            where profile='iincho'""").fetchone()[0]==0
        status=live.status(c)
        assert status["operation_observations"]==3
        assert all(r["nine_regions_persisted"] and r["three_integrators_persisted"]
                   for r in status["contexts"])
        assert not status["behavior_override"]
    again=live.sync(p)
    assert again["observed"]==again["scanned"]==0
    with world._connection(p) as c:
        assert c.execute("select count(*) from brain_9plus3_lived_events").fetchone()[0]==3

def test_incremental_once_and_protect_other_personas(tmp_path):
    p=_setup(tmp_path)
    live.sync(p)
    with world._connection(p) as c:
        c.execute("""insert into world_events(ts,profile,place,action,detail)
           values(?,?,?,?,?)""",(1005.,"iincho_n","garden","池のまわりを歩く","observed"))
        c.execute("""insert into world_events(ts,profile,place,action,detail)
           values(?,?,?,?,?)""",(1006.,"daino","garden","ゲームで遊ぶ","other"))
    result=live.sync(p)
    assert result["observed"]==1 and result["skipped"]==0
    with world._connection(p) as c:
        assert c.execute("select count(*) from brain_9plus3_lived_events").fetchone()[0]==4
        assert c.execute("""select count(*) from brain_9plus3_lived_events
          where profile!='iincho_n'""").fetchone()[0]==0

def test_conflicting_source_rolls_back_cursor_and_states(tmp_path):
    p=_setup(tmp_path)
    live.sync(p)
    with world._connection(p) as c:
        expected_last=c.execute("select last_world_event_id from brain_9plus3_live_cursor").fetchone()[0]
        c.execute("""insert into world_events(ts,profile,place,action,detail)
           values(?,?,?,?,?)""",(1010.,"iincho_n","garden","原っぱを走る","new"))
        c.execute("""insert into brain_9plus3_lived_events(
          profile,context_key,source_id,source_kind,action,operational_applied,
          task_success,ts,region_json,integrators_json,links_json)
          values(?,?,?,?,?,?,?,?,?,?,?)""",
          ("iincho_n","operation:rest",
           "world_events#"+str(c.execute("select max(id) from world_events").fetchone()[0]),
           "verified_operational_action","rest",1,None,1010.,"{}","{}","[]"))
    with pytest.raises(ValueError,match="conflicting"):
        live.sync(p)
    with world._connection(p) as c:
        assert c.execute("select last_world_event_id from brain_9plus3_live_cursor").fetchone()[0]==expected_last
        assert c.execute("""select count(*) from brain_9plus3_region_states
          where profile='iincho_n'""").fetchone()[0]==27

def test_backward_compatible_uninitialized_state(tmp_path):
    p=_setup(tmp_path)
    with world._connection(p) as c:
        state=live.status(c)
        assert not state["enabled"]
        assert state["operation_observations"]==0
