"""Guard v2: two disjoint windows, no short-lived false alarms."""
import brain_9plus3_strategy_guard_v2 as guard
import brain_9plus3_self_revision as revision
import brain_9plus3_task_learning as task
import world_space as world
import world_growth
from test_brain_9plus3_self_revision import sandbox,seed_task,CONTEXT

def _add(c,db,t,success,action="play"):
    assert world_growth.apply(db,"iincho_n",
         {"action":action,"name":"unit","reason":"unit_fixture"},at=t)["ok"]
    e=c.execute("select id from world_events where profile='iincho_n' and ts=? order by id desc limit 1",(t,)).fetchone()
    assert e
    task.record(c,"iincho_n",t,CONTEXT,action,int(e["id"]),success,
                source_kind="verified_task_evaluator",
                evaluator_id="sandbox_test",predicted_prior=.5)

def _approved(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        for i in range(12):
            _add(c,db,t0+i*3600,True if i%2==0 else False,
                 "play" if i%2==0 else "wander")
        for i in range(6):
            _add(c,db,t0+(12+i)*3600,True,"play")
        at=t0+18*3600
        pid=revision.propose(c,"iincho_n",CONTEXT,at=at)
        assert pid
        import self_growth
        self_growth.ensure(c)
        r=c.execute("insert into self_growth_reviews(profile,day,ts,status) values(?,?,?,'reserved')",
                 ("iincho_n","2000-01-01",at+1))
        assert revision.decide(c,"iincho_n",r.lastrowid,"adopt",
                        proposal_id=pid,at=at+1)["status"]=="adopt"
    return db,at+1

def test_no_approved_strategy(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        assert guard.assess(c,"iincho_n",CONTEXT,t0) is None

def test_single_failure_and_six_failures_do_not_trigger(tmp_path):
    db,t0=_approved(tmp_path)
    with world._connection(db) as c:
        for i in range(6):
            _add(c,db,t0+(i+1)*3600,False)
        result=guard.assess(c,"iincho_n",CONTEXT,t0+7*3600)
        assert result["can_apply_bias"],result
        assert result["baseline_n"]>=8
        assert result["earlier_window_success"] is None

def test_two_independent_windows_detect_long_decline(tmp_path):
    db,t0=_approved(tmp_path)
    with world._connection(db) as c:
        for i in range(12):
            _add(c,db,t0+(i+1)*3600,False)
        result=guard.assess(c,"iincho_n",CONTEXT,t0+13*3600)
        assert result["status"]=="contradicted",result
        assert result["earlier_window_success"]==0.
        assert result["recent_window_success"]==0.
        assert len(result["evidence_sources"])==12
        assert revision.current(c,"iincho_n",CONTEXT,at=t0+13*3600)

def test_fresh_results_keep_long_approved_strategy_active(tmp_path):
    db,t0=_approved(tmp_path)
    with world._connection(db) as c:
        for i in range(10):
            at=t0+(i+1)*86400
            _add(c,db,at,True)
        res=guard.assess(c,"iincho_n",CONTEXT,t0+10*86400+1)
        assert res["status"]=="active",res
        assert res["stale_days"]<1

def test_idle_strategy_expires_but_personal_memories_remain(tmp_path):
    db,t0=_approved(tmp_path)
    with world._connection(db) as c:
        res=guard.assess(c,"iincho_n",CONTEXT,t0+7*86400+1)
        assert res["status"]=="stale_evidence"
        assert revision.current(c,"iincho_n",CONTEXT,at=t0+7*86400+1)
        assert guard.guarded_bias(revision.bias,c,"iincho_n",CONTEXT,
           t0+7*86400+1,{"play":.6,"rest":.4})["weights"]=={"play":.6,"rest":.4}