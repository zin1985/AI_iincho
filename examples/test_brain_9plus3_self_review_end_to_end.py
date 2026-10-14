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