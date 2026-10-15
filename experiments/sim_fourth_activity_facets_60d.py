"""60-day paired simulated activity experience coverage for experiment-four.

Only aggregate action labels/frequencies are read from N, not original
autobiographical world-event rows, private memories or event detail.
Synthetic timestamps, source IDs and world ownership are all sim_fourth.
This evaluates observable information coverage and continuity, not success
rate or subjective emotion.
"""
from __future__ import annotations
import sqlite3,json,random,hashlib,collections,statistics
from pathlib import Path
import brain_9plus3_lived_learning as lived
import brain_9plus3_activity_facets as facet
from n_nine_three_lived_sync import ALLOW

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/simulations/fourth_activity_facets_60d"
WHO="sim_fourth"
T0=1792040000.
def frequency_distribution():
    source=ROOT/"data/daino.sqlite3"
    with sqlite3.connect("file:"+source.as_posix()+"?mode=ro",uri=True) as c:
        rows=c.execute("""select action,count(*) from world_events
          where profile='iincho_n' group by action order by action""").fetchall()
    assert sum(n for _,n in rows)>=486
    assert sum(n for a,n in rows if a in ALLOW)==178
    assert sum(n for a,n in rows if a in facet.LABEL_TO_CATEGORY)>=308
    assert all(a in ALLOW or a in facet.LABEL_TO_CATEGORY for a,n in rows)
    # This is aggregate vocabulary only; no N individual world events.
    return rows

def sequence(rows,seed,shift):
    rng=random.Random(seed)
    first=[a for label,n in rows for a in [label]*n]
    rng.shuffle(first)
    later=list(first)
    batch=len(first)
    if shift:
        labels=[a for a,n in rows]
        weights=[n*({"craft":3.8,"reading":2.2,"ideation":2.1,
              "reflection":2.1,"organization":2.1,"observation":.45,
              "recovery":.35,"movement":.45,"daily_routine":.7}.get(
               facet.LABEL_TO_CATEGORY.get(a),.6) if
               a not in ALLOW else .75) for a,n in rows]
        later=rng.choices(labels,weights=weights,k=batch)
    else:
        rng.shuffle(later)
    # 486 events in first 30 days, 486 in second 30 days.
    schedule=[]
    for phase,labels in enumerate((first,later)):
        for i,label in enumerate(labels):
            at=T0+phase*30*86400+(i+.5)*30*86400/batch
            schedule.append((at,label,phase))
    return schedule

def state_snapshot(c):
    table={r["name"] for r in c.execute("select name from sqlite_master where type='table'")}
    if "brain_9plus3_lived_events" not in table:
        return {"regions":0,"integrators":0}
    states={}
    for group,field in (("regions","brain_9plus3_region_states"),
                        ("integrators","brain_9plus3_integrator_states")):
        states[group]=c.execute("select count(*) from "+field+
             " where profile=?",(WHO,)).fetchone()[0]
    return states

def review_focus(c,at):
    """Factual attention focus, not an acted-on goal or emotional preference."""
    table={x["name"] for x in c.execute("select name from sqlite_master where type='table'")}
    if facet.TABLE not in table:return None
    candidates=c.execute("""select activity_category,count(*) n from
      brain_9plus3_activity_evidence
      where profile=? and ts>=? and ts<=? group by activity_category""",
      (WHO,at-3*86400,at)).fetchall()
    scores={}
    for row in candidates:
        name=row["activity_category"]
        engines=lived.snapshot(c,WHO,"activity:"+name)["integrators"]
        if not all(v is not None for v in engines.values()):continue
        count=int(row["n"])
        scores[name]=round(count*(.50*engines["reward_value"]+
           .25*engines["world_causal_model"]+
           .25*engines["meta_reflection"]),5)
    return max(scores,key=lambda a:(scores[a],a)) if scores else None

def arm(target,events,extended):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():target.unlink()
    c=sqlite3.connect(target)
    c.row_factory=sqlite3.Row
    try:
        c.execute("""create table world_events(
          id integer primary key,profile text,ts real,action text)""")
        lived.ensure(c)
        accepted=0;facets=0
        days=[]
        current_day=-1
        for ts,label,phase in events:
            day=int((ts-T0)//86400.)
            if current_day!=-1 and day!=current_day:
                if extended:facet.sync_connection(c,profile=WHO,limit=1500)
                days.append({"day":current_day,"focus":review_focus(c,T0+(current_day+1)*86400.-1),
                      "state_count":state_snapshot(c)})
            current_day=day
            cur=c.execute("insert into world_events(profile,ts,action) values(?,?,?)",
                          (WHO,ts,label))
            eid=int(cur.lastrowid)
            if label in ALLOW:
                act=ALLOW[label]
                lived.observe_operational(c,WHO,"operation:"+act,act,ts,
                     "world_events#"+str(eid),True)
                accepted+=1
        if extended:facet.sync_connection(c,profile=WHO,limit=1500)
        days.append({"day":current_day,"focus":review_focus(c,T0+(current_day+1)*86400.-1),
                      "state_count":state_snapshot(c)})
        total=c.execute("select count(*) from brain_9plus3_lived_events where profile=?",(WHO,)).fetchone()[0]
        status=facet.status(c,WHO)
        facets=status["activity_observations"]
        assert total==accepted+facets
        assert c.execute("""select count(*) from brain_9plus3_lived_events
          where task_success is not null""").fetchone()[0]==0
        assert c.execute("""select count(*) from brain_9plus3_goal_choices""").fetchone()[0]==0
        assert c.execute("""select count(*) from brain_9plus3_lived_events
          where profile!='sim_fourth'""").fetchone()[0]==0
        assert all(x["focus"] is None for x in days) if not extended else any(x["focus"] is not None for x in days)
        kinds={r["source_kind"] for r in c.execute("select distinct source_kind from brain_9plus3_lived_events")}
        assert kinds=={"verified_operational_action"}
        by_first=c.execute("""select count(*) from brain_9plus3_lived_events
              where profile=? and ts<?""",(WHO,T0+30*86400)).fetchone()[0]
        dups=c.execute("""select count(*)-count(distinct source_id)
          from brain_9plus3_lived_events""").fetchone()[0]
        assert dups==0
        summary={"recorded":total,"core":accepted,"new_facets":facets,
          "first_30d":by_first,"categories":status["categories"],
          "state_rows":state_snapshot(c),
          "focus_first":days[0]["focus"],
          "focus_last":days[-1]["focus"],
          "focus_days":sum(x["focus"] is not None for x in days),
          "focus_transitions":sum(a["focus"]!=b["focus"] for a,b in zip(days,days[1:])),
          "focus_first_half":collections.Counter(x["focus"] for x in days[:30] if x["focus"] is not None),
          "focus_second_half":collections.Counter(x["focus"] for x in days[30:] if x["focus"] is not None),
          "independent_task_grades":0,"behavior_changes":0,
          "no_N_persona_rows":True,"world_events":len(events)}
        # Persist source rows and all 9+3 state before another process
        # reopens this isolated DB for speech/reflection simulation.
        c.commit()
        with sqlite3.connect("file:"+target.as_posix()+"?mode=ro",uri=True) as check:
            n=check.execute("select count(*) from world_events where profile=?",(WHO,)).fetchone()[0]
            m=check.execute("select count(*) from brain_9plus3_lived_events where profile=?",(WHO,)).fetchone()[0]
        assert n==len(events) and m==summary["recorded"],"sandbox simulation not durably persisted"
        return summary
    finally:c.close()

def main(seeds=(991,992,993)):
    rows=frequency_distribution()
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in seeds:
        for change in (False,True):
            events=sequence(rows,seed,change)
            case=OUT/f"{seed}_{'shift30' if change else 'stationary'}"
            old=arm(case.with_name(case.name+"_baseline.sqlite3"),events,False)
            new=arm(case.with_name(case.name+"_extended.sqlite3"),events,True)
            assert old["recorded"]+new["new_facets"]==new["recorded"]
            assert old["first_30d"]==sum(n for a,n in rows if a in ALLOW)
            assert new["first_30d"]==sum(n for a,n in rows)
            assert new["new_facets"]>0 and new["state_rows"]["regions"]>=27+81
            assert new["state_rows"]["integrators"]>=9+27
            records.append({"seed":seed,"scenario":"shift30" if change else "stationary",
                    "baseline":old,"extended":new,
                    "first_30d_additional":new["first_30d"]-old["first_30d"],
                    "new_info_scopes":len(new["categories"])})
            print("FINISHED",seed,"drift",change,"coverage",old["recorded"],new["recorded"],
                  "new contexts",len(new["categories"]),"focus_changes",new["focus_transitions"],flush=True)
    p=OUT/"audit.json"
    report={"schema":"fourth_activity_facets_60d_v1","days":60,
      "aggregate_N_activity_distribution_no_autobiographical_memory":True,
      "synthetic_persona":"sim_fourth","seed_results":records,
      "baseline_action_outcomes":0,"extended_action_outcomes":0,
      "objective_success_gain_not_measured":True,
      "real_N_untouched":True}
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=dict),encoding="utf8")
    print("AUDIT",p,flush=True)
if __name__=="__main__":main()
