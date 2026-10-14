"""Read-only real N action-event shadow replay into an isolated nine-region DB.

All original world events are read-only and only a small, exact allowlist of
action names is mapped to objective *action occurrence*, not task success.
No authored goals, feelings, memories, or production sqlite are modified.
"""
from __future__ import annotations
import sqlite3,json,datetime
from pathlib import Path
import brain_9plus3_lived_learning as lived

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/daino.sqlite3"
TARGET=ROOT/"reports/simulations/n_actual_world_action_shadow"
N="iincho_n"
ALLOW={
    "ひと息つく":"rest",
    "静かに休む":"rest",
    "休息":"rest",
    "休む":"rest",
    "小道を散歩する":"wander",
    "原っぱを走る":"wander",
    "池のまわりを歩く":"wander",
    "散歩する":"wander",
    "ゲームで遊ぶ":"play",
    "積み木で遊ぶ":"play",
}
def main():
    if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
    TARGET.mkdir(parents=True,exist_ok=True)
    output=TARGET/"operational_only.sqlite3"
    count={"mapped":0,"skipped":0,"duplicate":0,"by_action":{}}
    # Read-only source cursor. Never attach or open production writable.
    with sqlite3.connect("file:"+SOURCE.as_posix()+"?mode=ro",uri=True,timeout=8.) as src:
        src.row_factory=sqlite3.Row
        events=src.execute("""select id,ts,profile,action from world_events
                      where profile=? order by ts,id""",(N,)).fetchall()
    with sqlite3.connect(output) as dst:
        dst.row_factory=sqlite3.Row
        lived.ensure(dst)
        for row in events:
            name=row["action"]
            category=ALLOW.get(name)
            if category is None:
                count["skipped"]+=1
                continue
            # We observed an event, not independently graded task success.
            event=lived.observe_operational(dst,N,"operation:"+category,category,
                        float(row["ts"]),"world_events#"+str(row["id"]),True)
            if event.get("duplicate"):
                count["duplicate"]+=1
            else:
                count["mapped"]+=1
            count["by_action"][category]=count["by_action"].get(category,0)+1
        count["original_N_world_events"]=len(events)
        rows=dst.execute("""select region,context_key,events from
          brain_9plus3_region_states where profile=?""",(N,)).fetchall()
        count["n_region_state_rows"]=len(rows)
        count["n_integrator_state_rows"]=dst.execute("""select count(*)
          from brain_9plus3_integrator_states where profile=?""",(N,)).fetchone()[0]
        count["objective_task_grades"]=dst.execute("""select count(*) from
          brain_9plus3_lived_events where profile=? and task_success is not null""",(N,)).fetchone()[0]
        count["persisted_operational_events"]=dst.execute("""select count(*) from
          brain_9plus3_lived_events where profile=?""",(N,)).fetchone()[0]
        count["verified_distinct_sources"]=dst.execute("""select count(distinct source_id)
          from brain_9plus3_lived_events where profile=?""",(N,)).fetchone()[0]
        count["other_persona_events"]=dst.execute("""select count(*) from
          brain_9plus3_lived_events where profile<>?""",(N,)).fetchone()[0]
    assert count["objective_task_grades"]==0
    assert count["other_persona_events"]==0
    assert count["persisted_operational_events"]==count["verified_distinct_sources"]
    assert count["original_N_world_events"]==sum((count["mapped"],count["skipped"],count["duplicate"])) if count["duplicate"]==0 else count["original_N_world_events"]==count["mapped"]+count["skipped"]+count["duplicate"]
    report={"schema":"N_world_events_operational_shadow_v1",
      "production_sqlite_opened_read_only":True,"production_sqlite_written":False,
      "shadow_action_occurrence_not_task_success":True,
      "no_llm_calls":True,"no_subjective_feeling_claimed":True,
      "counts":count,"shadow_sqlite":str(output)}
    (TARGET/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="shadow_sqlite"},ensure_ascii=False))
    return report
if __name__=="__main__":
    main()
