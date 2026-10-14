"""End-to-end test: existing N daily review accepts sourced 9+3 revision."""
from test_brain_9plus3_self_revision import sandbox,seed_task,CONTEXT
import world_space as world
import self_growth
import brain_9plus3_self_revision as revisions

def test_daily_N_review_uses_no_extra_model_call_and_reviews_a_real_proposal(tmp_path):
    db,start=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,start)
    seen=[]
    def fake_model(who,payload,mode):
        seen.append((who,payload,mode))
        proposals=payload["9＋3の長期方針改訂案（採否は本人が決める）"]["pending_machine_proposals"]
        assert len(proposals)==1
        assert proposals[0]["verified_sources"]
        assert proposals[0]["not_N_written"] is True
        return ({"reflection":"これは試験用にスタブが返す文章",
                 "strategy_review":{"proposal_id":proposals[0]["id"],
                                    "decision":"adopt",
                                    "reason":"テスト専用。出典で実績を確認した"}},
                "stub-N-model")
    outcome=self_growth.review_once(db,{},at=at+2,call_fn=fake_model,
                                     force=True,selected="iincho_n")
    assert outcome["status"]=="completed",outcome
    assert len(seen)==1
    assert seen[0][0]=="iincho_n"
    with world._connection(db) as c:
        strategies=revisions.current(c,"iincho_n",CONTEXT,at=at+3)
        assert strategies and strategies[0]["action"]=="play"
        assert "persona_model_daily_review" in strategies[0]["review_source"]
        decisions=c.execute("""select decision from brain_9plus3_strategy_decisions
                               where profile='iincho_n'""").fetchall()
        assert len(decisions)==1 and decisions[0]["decision"]=="adopt"

def test_daily_review_authored_value_enters_existing_self_model_and_next_features(tmp_path):
    import small_brain_goal_learning as learn
    db,start=sandbox(tmp_path)
    with world._connection(db) as c:
        self_growth.ensure(c)
        at=seed_task(c,db,start)
        event=c.execute("""select id from brain_9plus3_task_outcomes
                        where profile='iincho_n' order by id desc limit 1""").fetchone()
        proof_id=event["id"]
        before=learn.snapshot(c,"iincho_n",at+1,(1.,0.,0.,0.,0.),None)
    def fake_review(who,payload,mode):
        return ({"reflection":"仮想試行の結果に基づき判断を見直す",
          "value":{"name":"失敗を振り返り別の方法を試す",
                   "interpretation":"行動を繰り返すだけでなく、課題の結果を見て方法を更新したい",
                   "evidence":"brain_9plus3_task_outcomes#"+str(proof_id)}},
          "stub-self-review")
    r=self_growth.review_once(db,{},at=at+2,call_fn=fake_review,
                               force=True,selected="iincho_n")
    assert r["status"]=="completed",r
    assert r["value_candidate"] is True
    with world._connection(db) as c:
        values=c.execute("""select name,interpretation,source_kind from
            self_growth_values where profile='iincho_n'""").fetchall()
        assert len(values)==1
        assert values[0]["source_kind"]=="daily_review"
        evidence=c.execute("""select source_kind,source_id
            from self_growth_review_evidence where candidate_type='value'""").fetchone()
        assert evidence["source_kind"]=="brain_9plus3_task_outcomes"
        assert evidence["source_id"]==proof_id
        after=learn.snapshot(c,"iincho_n",at+3,(1.,0.,0.,0.,0.),None)
        assert before["features"]!=after["features"]