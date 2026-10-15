"""Production N: evidence-only persistent nine-region and three-integrator sync.

Source: world_events in the SAME live SQLite transaction.
No task grades inferred. No simulated 4th-persona source, authored feeling,
memory, goal choice, approved strategy, or action probability is changed.
"""
from __future__ import annotations
import json,sqlite3
from pathlib import Path

import brain_9plus3_lived_learning as lived

PROFILE="iincho_n"
# Deliberately conservative list, from validated sandbox shadow replay.
ALLOW={
    "ひと息つく":"rest","静かに休む":"rest","休息":"rest","休む":"rest",
    "小道を散歩する":"wander","原っぱを走る":"wander",
    "池のまわりを歩く":"wander","散歩する":"wander",
    "ゲームで遊ぶ":"play","積み木で遊ぶ":"play",
}
VERSION="nine_three_n_operational_only_v1"
TABLE="brain_9plus3_live_cursor"

def ensure(c):
    lived.ensure(c)
    c.execute("""create table if not exists brain_9plus3_live_cursor(
        profile text primary key, last_world_event_id integer not null,
        scanned_events integer not null, observed_events integer not null,
        skipped_events integer not null, updated_at real not null)""")

def sync_connection(c, *, limit=1200):
    """Incrementally follow N's world actions. Caller owns the transaction.

    Works while N is active or asleep, without forcing a new action or LLM.
    """
    ensure(c)
    checkpoint=c.execute("select last_world_event_id,scanned_events,observed_events,skipped_events "
                         "from "+TABLE+" where profile=?",(PROFILE,)).fetchone()
    last=int(checkpoint["last_world_event_id"]) if checkpoint else 0
    old_scanned=int(checkpoint["scanned_events"]) if checkpoint else 0
    old_seen=int(checkpoint["observed_events"]) if checkpoint else 0
    old_skipped=int(checkpoint["skipped_events"]) if checkpoint else 0
    limit=max(1,min(int(limit),2000))
    rows=c.execute("""select id,ts,profile,action from world_events
        where id>? and profile=? order by id limit ?""",
        (last,PROFILE,limit)).fetchall()
    observed=0
    skipped=0
    for row in rows:
        if row["profile"]!=PROFILE or not isinstance(row["action"],str):
            raise ValueError("invalid source profile or action")
        action=ALLOW.get(row["action"])
        if action is None:
            skipped+=1
            continue
        source="world_events#"+str(int(row["id"]))
        # Observations, not independent success grades.
        existing=c.execute("""select profile,action,ts,task_success,operational_applied
            from brain_9plus3_lived_events where source_id=?""",(source,)).fetchone()
        if existing is not None:
            if (existing["profile"]!=PROFILE or existing["action"]!=action or
                abs(float(existing["ts"])-float(row["ts"]))>1e-5 or
                existing["task_success"] is not None or
                existing["operational_applied"]!=1):
                raise ValueError("conflicting historical observation")
            continue
        inserted=lived.observe_operational(c,PROFILE,"operation:"+action,
                         action,float(row["ts"]),source,True)
        if not inserted["duplicate"]:
            observed+=1
    if rows:
        end=int(rows[-1]["id"])
        c.execute("""insert into brain_9plus3_live_cursor(
           profile,last_world_event_id,scanned_events,observed_events,skipped_events,updated_at)
           values(?,?,?,?,?,?) on conflict(profile) do update set
           last_world_event_id=excluded.last_world_event_id,
           scanned_events=excluded.scanned_events,
           observed_events=excluded.observed_events,
           skipped_events=excluded.skipped_events,
           updated_at=excluded.updated_at""",
           (PROFILE,end,old_scanned+len(rows),old_seen+observed,
            old_skipped+skipped,float(rows[-1]["ts"])))
    return {"version":VERSION,"scanned":len(rows),
            "observed":observed,"skipped":skipped,
            "last_world_event_id":int(rows[-1]["id"]) if rows else last,
            "grades_fabricated":0,"llm_requests":0,
            "memory_or_goal_changed":False}

def sync(db_path,*,limit=1200):
    import world_space
    c=world_space._connection(Path(db_path))
    try:
        c.execute("begin immediate")
        result=sync_connection(c,limit=limit)
        c.execute("commit")
        return result
    except Exception:
        c.execute("rollback")
        raise
    finally:
        c.close()

def status(c):
    has=c.execute("select 1 from sqlite_master where type='table' and name=?",(TABLE,)).fetchone()
    if not has:
        return {"enabled":False,"operation_observations":0,
                "independently_graded":0,"behavior_override":False}
    cursor=c.execute("select * from "+TABLE+" where profile=?",(PROFILE,)).fetchone()
    if not cursor:
        return {"enabled":True,"operation_observations":0,
                "independently_graded":0,"behavior_override":False}
    summaries=[]
    for name in ("rest","wander","play"):
        context="operation:"+name
        row=c.execute("""select count(*) as n from brain_9plus3_lived_events
            where profile=? and context_key=? and task_success is null""",
            (PROFILE,context)).fetchone()
        if row["n"]:
            s=lived.snapshot(c,PROFILE,context)
            summaries.append({"operation":name,"observations":int(row["n"]),
                  "latest_sources":s["sources"][:3],
                  "nine_regions_persisted":s["all_nine_defined"],
                  "three_integrators_persisted":all(v is not None
                      for v in s["integrators"].values())})
    return {"enabled":True,"version":VERSION,
            "operation_observations":int(cursor["observed_events"]),
            "source_events_scanned":int(cursor["scanned_events"]),
            "unclassified":int(cursor["skipped_events"]),
            "contexts":summaries,
            "independently_graded":0,
            "behavior_override":False,
            "not_persona_feelings":True,
            "is_world_action_not_task_success":True}
