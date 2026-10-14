"""Long-term N self-revision requires an actual review and audited task data."""
import json
import world_space as world
import world_growth
import pytest
import self_growth
import brain_9plus3_task_learning as task
import brain_9plus3_self_revision as revision
import brain_9plus3_policy_bridge as policy

CONTEXT="project:strategy_unit_test"
BASE={"rest":.4,"wander":.35,"play":.25}

def sandbox(tmp_path):
    db=tmp_path/"fourth_safe.sqlite3"
    world.snapshot(db)
    return db,1791900000.

def seed_task(c,db,t0):
    # Test fixture: 12 isolated synthetic world actions stored in the
    # *test DB only*. N's actual production history is never touched.
    for i in range(12):
        act="play" if i%2==0 else "wander"
        timestamp=t0+i*3600
        result=world_growth.apply(db,"iincho_n",
            {"action":act,"name":"test_task","reason":"temporary_test_fixture"},at=timestamp)
        assert result["ok"]
        event=c.execute("""select id from world_events where profile=? and ts=?
           order by id desc limit 1""",("iincho_n",timestamp)).fetchone()
        assert event is not None
        task.record(c,"iincho_n",timestamp,CONTEXT,act,int(event["id"]),
                    act=="play",source_kind="verified_task_evaluator",
                    evaluator_id="test_case_only",predicted_prior=.5)
    return t0+12*3600

def test_proposal_is_not_an_automatic_personal_preference(tmp_path):
    db,now=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        ready=seed_task(c,db,now)
        proposal=revision.propose(c,"iincho_n",CONTEXT,at=ready)
        assert isinstance(proposal,int)
        assert revision.propose(c,"iincho_n",CONTEXT,at=ready)==proposal
        pending=revision.pending(c,"iincho_n",at=ready)
        assert pending[0]["id"]==proposal
        assert pending[0]["approved"] is False
        assert pending[0]["suggested_action"]=="play"
        assert revision.current(c,"iincho_n",at=ready)==[]
        paired=policy.counterfactual(c,"iincho_n",ready,None,BASE,task_context=CONTEXT)
        assert paired["evidence"].get("approved_persona_strategy_sources")==[]
        assert paired["evidence"]["strategy_has_persona_authorization"] is False

def test_only_persona_approved_revision_changes_future_policy(tmp_path):
    db,start=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,start)
        proposal=revision.propose(c,"iincho_n",CONTEXT,at=at)
        baseline=policy.counterfactual(c,"iincho_n",at+.5,None,BASE,task_context=CONTEXT)
        bad=revision.decide(c,"iincho_n",None,"adopt",
            proposal_id=proposal,at=at+.5)
        assert bad["status"]=="unauthorized"
        assert revision.current(c,"iincho_n",at=at+.5)==[]
        review=c.execute("""insert into self_growth_reviews(
            profile,day,ts,status) values(?,?,?,'reserved')""",
            ("iincho_n","2026-10-12",at+1))
        result=revision.decide(c,"iincho_n",review.lastrowid,"adopt",
             proposal_id=proposal,reason="実際に成績を確認して試す",at=at+1)
        assert result["status"]=="adopt"
        assert result["self_model_changed"] is True
        next_policy=policy.counterfactual(c,"iincho_n",at+2,None,BASE,
                                          task_context=CONTEXT)
        assert next_policy["integrated"]["play"]>baseline["integrated"]["play"]
        assert next_policy["evidence"]["approved_persona_strategy_sources"]
        assert next_policy["evidence"]["strategy_has_persona_authorization"]
        assert revision.decide(c,"iincho_n",review.lastrowid,"adopt",
             proposal_id=proposal,at=at+1)["status"]=="already_decided"
        different=policy.counterfactual(c,"iincho_n",at+2,None,BASE,
              task_context="scenario:unrelated")
        assert not different["evidence"].get("approved_persona_strategy_sources")
        assert revision.current(c,"iincho",at=at+2)==[]

def test_reject_and_defer_never_write_approved_strategy(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,t0)
        id=revision.propose(c,"iincho_n",CONTEXT,at=at)
        review=c.execute("""insert into self_growth_reviews(profile,day,ts,status)
          values(?,?,?,'reserved')""",("iincho_n","2026-10-13",at+1))
        assert revision.decide(c,"iincho_n",review.lastrowid,"defer",
            proposal_id=id,at=at+1)["status"]=="defer"
        assert revision.current(c,"iincho_n")==[]
        assert not revision.pending(c,"iincho_n",at=at+2)
        assert revision.decide(c,"iincho_n",review.lastrowid,"adopt",
            proposal_id=id,at=at+1)["status"]=="already_decided"

def test_no_synthetic_persona_authorization_for_N(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,t0)
        id=revision.propose(c,"iincho_n",CONTEXT,at=at)
        assert revision.decide(c,"iincho_n",123,"adopt",proposal_id=id,
                 synthetic_test_only=True,at=at+.1)["status"]=="test_profile_required"
        assert revision.decide(c,"iincho_n",123,"adopt",proposal_id=id,
                 at=at+.1)["status"]=="unverified_review"
        assert revision.current(c,"iincho_n")==[]

def test_review_proof_accepts_only_same_person_verified_task(tmp_path):
    db,t0=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,t0)
        row=c.execute("select id from brain_9plus3_task_outcomes where profile='iincho_n' limit 1").fetchone()
        proof=self_growth._review_proof(c,"iincho_n",
                    "brain_9plus3_task_outcomes#"+str(row["id"]))
        assert proof["source_id"]==row["id"]
        assert self_growth._review_proof(c,"iincho",
                    "brain_9plus3_task_outcomes#"+str(row["id"])) is None
        assert self_growth._review_proof(c,"iincho_n",
                    "brain_9plus3_task_outcomes#99999999") is None
        assert self_growth._review_proof(c,"iincho_n",
                    "brain_9plus3_task_outcomes#bad") is None