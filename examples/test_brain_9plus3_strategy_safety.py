"""Safety of stale 9+3 strategy evaluation, independent of production wiring."""
import brain_9plus3_strategy_safety as safety
import brain_9plus3_self_revision as revisions
import brain_9plus3_task_learning as tasks
import self_growth
import world_growth
import world_space as world
from test_brain_9plus3_self_revision import seed_task,sandbox,CONTEXT,BASE

def approved(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,t0)
        pid=revisions.propose(c,"iincho_n",CONTEXT,at=at)
        r=c.execute("""insert into self_growth_reviews(profile,day,ts,status)
           values(?,?,?,'reserved')""",("iincho_n","test",at+1))
        yes=revisions.decide(c,"iincho_n",r.lastrowid,"adopt",
                          proposal_id=pid,at=at+1,reason="verified")
        assert yes["status"]=="adopt"
    return db,at+1

def append_failed(db,begin,n):
    with world._connection(db) as c:
        for i in range(n):
            at=begin+(i+1)*3600
            result=world_growth.apply(db,"iincho_n",
                {"action":"play","name":"test","reason":"test_failure"},at=at)
            assert result["ok"]
            row=c.execute("""select id from world_events
               where profile='iincho_n' and ts=?
               order by id desc limit 1""",(at,)).fetchone()
            assert row
            tasks.record(c,"iincho_n",at,CONTEXT,"play",int(row["id"]),False,
                source_kind="verified_task_evaluator",
                evaluator_id="unit_test_declining_task",predicted_prior=.8)

def test_without_approval_does_not_flag(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        assert safety.assess(c,"iincho_n",CONTEXT,at=t0) is None
        assert safety.monitor(c,"iincho_n",at=t0)==[]

def test_fresh_approved_strategy_is_active_and_other_task_absent(tmp_path):
    db,at=approved(tmp_path)
    with world._connection(db) as c:
        s=safety.assess(c,"iincho_n",CONTEXT,at=at+1)
        assert s["status"]=="active"
        assert s["safe_to_apply_bias"]
        assert not s["decline_detected"]
        assert safety.assess(c,"iincho_n","scenario:another",at=at+1) is None

def test_declining_results_trigger_review_without_rewriting_persona(tmp_path):
    db,at=approved(tmp_path)
    append_failed(db,at,16)
    with world._connection(db) as c:
        s=safety.assess(c,"iincho_n",CONTEXT,at=at+17*3600)
        assert s["status"]=="needs_review",s
        assert s["decline_detected"] is True
        assert len(s["evidence_sources"])>=4
        assert not s["safe_to_apply_bias"]
        assert len(revisions.current(c,"iincho_n",CONTEXT,at=at+17*3600))==1
        assert safety.monitor(c,"iincho_n",at=at+17*3600)
        assert safety.decide(c,"iincho_n",None,CONTEXT,"revoke",
                             at=at+17*3600)["status"]=="unauthorized"
        assert len(revisions.current(c,"iincho_n",CONTEXT,at=at+17*3600))==1

def test_review_can_revoke_numeric_bias_without_deleting_personal_memory(tmp_path):
    db,at=approved(tmp_path)
    append_failed(db,at,16)
    with world._connection(db) as c:
        now=at+17*3600
        review=c.execute("""insert into self_growth_reviews(profile,day,ts,status)
         values(?,?,?,'reserved')""",("iincho_n","test2",now))
        result=safety.decide(c,"iincho_n",review.lastrowid,CONTEXT,
                              "revoke",at=now,reason="客観的な不振")
        assert result["status"]=="revoke"
        assert result["numeric_bias_revoked"]
        assert not result["memories_deleted"]
        assert revisions.current(c,"iincho_n",CONTEXT,at=now)==[]
        past=c.execute("select count(*) from brain_9plus3_strategy_decisions").fetchone()[0]
        assert past==1
        audited=c.execute("select count(*) from brain_9plus3_strategy_recheck_decisions").fetchone()[0]
        assert audited==1

def test_strategy_expires_after_seven_days_without_erasing_record(tmp_path):
    db,at=approved(tmp_path)
    with world._connection(db) as c:
        state=safety.assess(c,"iincho_n",CONTEXT,at=at+7*86400+.01)
        assert state["status"]=="expired"
        assert not state["safe_to_apply_bias"]
        assert revisions.current(c,"iincho_n",CONTEXT,at=at+7*86400+.01)