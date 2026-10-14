"""N shadow ingest is append-only, deterministic and never grades success."""
import sqlite3,hashlib,json
from pathlib import Path
from tools import replay_n_actual_world_actions_shadow as shadow

def _create(path):
    with sqlite3.connect(path) as c:
        c.execute("""create table world_events(id integer primary key,
             profile text, ts real, action text)""")
        c.executemany("""insert into world_events(profile,ts,action)
              values(?,?,?)""",[
            ("iincho_n",100000.,"小道を散歩する"),
            ("iincho_n",101000.,"ゲームで遊ぶ"),
            ("iincho_n",102000.,"静かに休む"),
            ("iincho_n",103000.,"曖昧な行動は分類しない"),
            ("sim_fourth",104000.,"ゲームで遊ぶ")])

def test_shadow_ingestion_and_source_integrity(tmp_path,monkeypatch):
    source=tmp_path/"source.sqlite3";_create(source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(shadow,"SOURCE",source)
    monkeypatch.setattr(shadow,"TARGET",tmp_path/"shadow")
    one=shadow.main()
    assert one["production_sqlite_opened_read_only"]
    assert one["counts"]["mapped"]==3
    assert one["counts"]["skipped"]==1
    assert one["counts"]["objective_task_grades"]==0
    assert one["counts"]["other_persona_events"]==0
    assert one["counts"]["n_region_state_rows"]==27
    assert one["counts"]["n_integrator_state_rows"]==9
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    twice=shadow.main()
    assert twice["counts"]["duplicate"]==3
    assert twice["counts"]["mapped"]==0
    assert twice["counts"]["persisted_operational_events"]==3
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
