"""Small brain region processors must produce sourced, non-subjective notes."""
from test_brain_9plus3_task_learning import make_db,evaluate
import world_space as world
import brain_9plus3_cognitive_loop as cog

BASE={"rest":.4,"wander":.4,"play":.2}
def test_nine_regions_three_integrators_and_empty_state(tmp_path):
    db,now=make_db(tmp_path)
    with world._connection(db) as c:
        result=cog.inspect(c,"iincho_n",now,"scenario:tool_problem","素材を試す",BASE)
        assert len(result["nine_regions"])==9
        assert len(result["three_integrators"])==3
        assert result["exploration_epsilon"]==.30
        assert result["reflective_note"]["potential_action"] is None
        assert result["reflective_note"]["unobserved_subjective_feelings"] is None
        assert result["nine_regions"]["hentou"]["felt_emotion"] is None
        assert result["nine_regions"]["naijima"]["felt_satisfaction"] is None
        assert not result["has_real_N_inner_dialogue"]

def test_region_processors_notice_conflict_and_use_only_verified_history(tmp_path):
    db,now=make_db(tmp_path)
    with world._connection(db) as c:
        for n in range(14):
            evaluate(c,db,now+n*3600,"rest",False)
        for n in range(7):
            evaluate(c,db,now+(n+14)*3600,"wander",True)
        state=cog.inspect(c,"iincho_n",now+22*3600,
                 "scenario:tool_problem","見直す",BASE)
        assert state["source_count"]==21
        assert state["nine_regions"]["kaiba"]["retrieved_events"]==21
        assert state["nine_regions"]["zentou"]["best_evidenced_action"]=="wander"
        assert state["nine_regions"]["taijou"]["habit_action"]=="rest"
        assert state["three_integrators"]["meta_reflection"]["should_reconsider"]
        assert state["reflective_note"]["verified_sources"]
        assert state["nine_regions"]["shounou"]["prediction_error_mean"] is not None
        # Wrong profile cannot access N's outcome sources.
        other=cog.inspect(c,"iincho",now+22*3600,
            "scenario:tool_problem","見直す",BASE)
        assert other["source_count"]==0

def test_9plus3_snapshot_does_not_invent_moods(tmp_path):
    db,now=make_db(tmp_path)
    with world._connection(db) as c:
        evaluate(c,db,now,"rest",False)
        snap=cog.inspect(c,"iincho_n",now+1,
                         "scenario:tool_problem","創作",BASE)
        assert snap["three_integrators"]["reward_value"]["intrinsic_felt_reward"] is None
        assert snap["nine_regions"]["shichoukashibu"]["physical_fatigue"] is None
        assert snap["nine_regions"]["toutai"]["N_authored_reflection"] is False
        assert "私" not in snap["reflective_note"]["observation"]
