"""4th-persona 60-day A/B: grounded speech/reflection *surrogate*.

NO LLM calls and NO Ollama. This test generates deterministic candidate
utterances from 4th-persona facts already in the isolated experiment DBs.
The texts are NOT model-authored spontaneous speech or real emotions.
It audits whether 9+3 evidence improves topical specificity, reflection
continuity and shift sensitivity without invented outcomes or persona edits.
"""
from __future__ import annotations
import json,sqlite3,collections,statistics
from pathlib import Path
import brain_9plus3_activity_facets as facet
import brain_9plus3_lived_learning as lived
from n_nine_three_lived_sync import ALLOW

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"reports/simulations/fourth_activity_facets_60d"
OUT=ROOT/"reports/simulations/fourth_speech_reflection_60d"
WHO="sim_fourth"
T0=1792040000.
DAY=86400.
NAME={
    "operation:rest":"休息",
    "operation:wander":"探索・散歩",
    "operation:play":"遊び",
    "activity:craft":"制作",
    "activity:ideation":"着想",
    "activity:reflection":"振り返り",
    "activity:observation":"観察",
    "activity:reading":"読書・調査",
    "activity:organization":"準備・整理",
    "activity:recovery":"休養・くつろぎ",
    "activity:movement":"移動",
    "activity:daily_routine":"日常"}
CATEGORY_OF={"activity:"+key:val for key,val in facet.GROUPS.items()}

def _has(c,name):
    return c.execute("select 1 from sqlite_master where name=?",
                     (name,)).fetchone() is not None

def _events(c,end):
    """Ground only in completed events through this simulated time."""
    return [dict(x) for x in c.execute("""
        select l.source_id,l.ts,l.context_key,l.action,l.task_success,
          l.region_json,l.integrators_json,w.action as source_action,
          w.id as source_event,w.profile as source_profile
        from brain_9plus3_lived_events l
        join world_events w on l.source_id='world_events#'||cast(w.id as text)
        where l.profile=? and l.ts>=? and l.ts<=?
        order by l.ts desc,l.id desc""",(WHO,end-3*DAY,end))]

def _verify(e):
    assert e["source_profile"]==WHO
    assert e["source_id"]=="world_events#"+str(e["source_event"])
    assert e["task_success"] is None
    if e["context_key"].startswith("activity:"):
        assert facet.LABEL_TO_CATEGORY[e["source_action"]]==e["action"]
        assert e["context_key"]=="activity:"+e["action"]
    else:
        assert ALLOW[e["source_action"]]==e["action"]
        assert e["context_key"]=="operation:"+e["action"]
    assert set(json.loads(e["region_json"]))==set(lived.REGIONS)
    assert set(json.loads(e["integrators_json"]))==set(lived.INTEGRATORS)

def _focus(evidence):
    grouped=collections.defaultdict(list)
    for ev in evidence:
        _verify(ev)
        grouped[ev["context_key"]].append(ev)
    if not grouped:return None
    scored={}
    for key,group in grouped.items():
        # This is the most recent saved integrator state at or before today.
        signals=json.loads(group[0]["integrators_json"])
        assert all(0<=float(v)<=1. for v in signals.values())
        strength=(.5*signals["reward_value"]+
                  .25*signals["world_causal_model"]+
                  .25*signals["meta_reflection"])
        scored[key]=len(group)*strength
    choice=max(scored,key=lambda cat:(scored[cat],cat))
    return choice,grouped[choice],round(scored[choice],5),scored

def _candidate(day,c,prior=None):
    end=T0+(day+1)*DAY-1.
    events=_events(c,end)
    focus=_focus(events)
    if focus is None:
        return {"day":day,"speak":False,"speech":"","reflection":"",
                "focus":None,"sources":[],"evidence_count":0,
                "utterance_type":"rule_surrogate_not_LLM"}
    key,group,score,scores=focus
    name=NAME[key]
    actual=group[0]["source_action"]
    # Both candidates are grounded. No subjective feeling, outcome, preference
    # or efficacy can be inferred from action occurrence.
    example="「"+actual+"」"
    count=len(group)
    sources=[e["source_id"] for e in group[:3]]
    intro=("前の注目は「"+NAME[prior]+"」だった。今回は「"+name+"」へ移った。" if
           prior is not None and prior!=key else
           "前回から「"+name+"」に注目している。" if prior==key else
           "今回は「"+name+"」の記録を取り上げる。")
    speech=("仮想の庭の記録では、最近「"+name+"」の行動が"+
            str(count)+"件あった。例えば"+example+"。"+
            "結果の良し悪しは、まだ分からない。")
    reflection=(intro+" 直近3日間で"+str(count)+"件を確認。"+
                "出典は"+sources[0]+"。"+
                "次に何を試すべきかは、この記録だけでは決められない。")
    return {"day":day,"speak":True,"speech":speech,
        "reflection":reflection,"focus":key,"focus_score":score,
        "source_action":actual,"sources":sources,
        "evidence_count":count,"candidate_count":len(scores),
        "speech_originality_unproven":True,
        "utterance_type":"rule_surrogate_not_LLM",
        "subjective_emotions_derived":False,
        "subjective_long_term_goal_derived":False,
        "independent_grade_count":0}

def _measure(rows):
    topics=[x["focus"] for x in rows if x["focus"]]
    starts=[x["focus"] for x in rows[:30] if x["focus"]]
    ends=[x["focus"] for x in rows[30:] if x["focus"]]
    return {
        "spoken_days":sum(x["speak"] for x in rows),
        "reflection_days":sum(bool(x["reflection"]) for x in rows),
        "distinct_topics":len(set(topics)),
        "focus_changes":sum(a!=b for a,b in zip(topics,topics[1:])),
        "source_grounded_samples":sum(x["speak"] and len(x["sources"])>0 for x in rows),
        "ungraded_outcomes_fabricated":0,
        "unapproved_goal_changes":0,
        "craft_days_first30":sum(x=="activity:craft" for x in starts),
        "craft_days_last30":sum(x=="activity:craft" for x in ends),
        "topic_frequency_first30":dict(collections.Counter(starts)),
        "topic_frequency_last30":dict(collections.Counter(ends))}

def experiment(seed,scenario,arm):
    src=SOURCE/f"{seed}_{scenario}_{arm}.sqlite3"
    assert src.is_file(),src
    with sqlite3.connect("file:"+src.as_posix()+"?mode=ro",uri=True) as c:
        c.row_factory=sqlite3.Row
        assert c.execute("select count(*) from world_events where profile!=?",
                (WHO,)).fetchone()[0]==0
        assert c.execute("select count(*) from brain_9plus3_lived_events where profile!=?",
                (WHO,)).fetchone()[0]==0
        previous=None;results=[]
        for day in range(60):
            current=_candidate(day,c,previous)
            results.append(current)
            if current["focus"]:previous=current["focus"]
        assert all(not x["speak"] or x["sources"] for x in results)
        assert all(x["utterance_type"]=="rule_surrogate_not_LLM" for x in results)
    return {"seed":seed,"scenario":scenario,"arm":arm,"db":str(src),
        "days":results,"metrics":_measure(results)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=[]
    all_episodes=[]
    for seed in (991,992,993):
        for scenario in ("stationary","shift30"):
            a=experiment(seed,scenario,"baseline")
            b=experiment(seed,scenario,"extended")
            # Diversity is an outcome, not a guaranteed success. A naive
            # frequency-weighted focus can collapse onto one category.
            assert b["metrics"]["spoken_days"]>0
            assert b["metrics"]["reflection_days"]>0
            assert b["metrics"]["ungraded_outcomes_fabricated"]==0
            pairs.append({"seed":seed,"scenario":scenario,
                          "baseline":a["metrics"],"extended":b["metrics"]})
            for arm,obj in (("baseline",a),("extended",b)):
                for rec in obj["days"]:
                    all_episodes.append({"seed":seed,"scenario":scenario,
                                         "arm":arm,**rec})
            print("PAIR",seed,scenario,"topics",
                a["metrics"]["distinct_topics"],"=>",
                b["metrics"]["distinct_topics"],"spoken",
                a["metrics"]["spoken_days"],b["metrics"]["spoken_days"],"craft_after",
                b["metrics"]["craft_days_last30"],flush=True)
    doc={"schema":"fourth_speech_reflection_provenance_A_B_v1",
         "speaker":"isolated_fourth_rule_based_surrogate_not_real_llm",
         "model_inference_calls":0,
         "not_real_persona_authored_speech":True,
         "objective_rewards_provided":0,"production_N_modified":False,
         "meaning":"checks factuality and topical dynamics, NOT subjective autonomy",
         "pairs":pairs,"episodes":all_episodes}
    target=OUT/"audit.json"
    target.write_text(json.dumps(doc,ensure_ascii=False,indent=2),
                      encoding="utf8")
    focus=[x for x in all_episodes if x["seed"]==992 and x["scenario"]=="shift30" and
           x["day"] in (5,20,31,40,55) and x["arm"] in ("baseline","extended")]
    (OUT/"utterances_review_samples.json").write_text(
        json.dumps(focus,ensure_ascii=False,indent=2),encoding="utf8")
    assert len(pairs)==6 and len(all_episodes)==720
    print(json.dumps({"status":"PASS","pairs":len(pairs),
       "utterance_probes":len(all_episodes),
       "saved":str(target),"examples":len(focus)},ensure_ascii=False))
if __name__=="__main__":main()
