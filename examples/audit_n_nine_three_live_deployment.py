"""Read-only after-deployment verification. Counts and joins only, no private data."""
import sqlite3,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/"data/daino.sqlite3"
with sqlite3.connect("file:"+p.as_posix()+"?mode=ro",uri=True,timeout=8) as c:
    c.row_factory=sqlite3.Row
    names={r["name"] for r in c.execute("select name from sqlite_master where type='table'")}
    assert {"brain_9plus3_lived_events","brain_9plus3_region_states",
            "brain_9plus3_integrator_states","brain_9plus3_live_cursor"}<=names
    obs=c.execute("""select count(*) from brain_9plus3_lived_events
          where profile='iincho_n'""").fetchone()[0]
    regions=c.execute("""select count(*) from brain_9plus3_region_states
          where profile='iincho_n'""").fetchone()[0]
    integrators=c.execute("""select count(*) from brain_9plus3_integrator_states
          where profile='iincho_n'""").fetchone()[0]
    task=c.execute("""select count(*) from brain_9plus3_lived_events
          where profile='iincho_n' and task_success is not null""").fetchone()[0]
    other=c.execute("""select count(*) from brain_9plus3_lived_events
          where profile!='iincho_n'""").fetchone()[0]
    unmatched=c.execute("""select count(*) from brain_9plus3_lived_events l
          left join world_events e on l.source_id='world_events#'||cast(e.id as text)
             and e.profile=l.profile and e.ts=l.ts
          where l.profile='iincho_n' and e.id is null""").fetchone()[0]
    accepted=c.execute("select count(*) from brain_9plus3_persona_strategies where profile='iincho_n'").fetchone()[0]
    draft=c.execute("select count(*) from brain_9plus3_strategy_proposals where profile='iincho_n'").fetchone()[0]
    outcome=c.execute("select count(*) from brain_9plus3_strategy_decisions where profile='iincho_n'").fetchone()[0]
    cursor=dict(c.execute("""select last_world_event_id,scanned_events,
          observed_events,skipped_events from brain_9plus3_live_cursor
          where profile='iincho_n'""").fetchone())
    assert regions==27 and integrators==9
    assert obs==cursor["observed_events"]
    assert task==0 and other==0 and unmatched==0
    assert accepted==draft==outcome==0
    assert cursor["scanned_events"]==cursor["observed_events"]+cursor["skipped_events"]
    result={"status":"verified","N_lived_events":obs,
         "N_region_states":regions,"N_integrator_states":integrators,
         "independent_task_grades":task,"non_N_rows":other,
         "source_mismatch":unmatched,"approved_strategies":accepted,
         "pending_strategy_proposals":draft,"strategy_decisions":outcome,
         "cursor":cursor}
    print(json.dumps(result,ensure_ascii=False))
