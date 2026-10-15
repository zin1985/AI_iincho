"""V7 objective-grade causal probe and per-task bounded stop tests."""
import sqlite3
from unittest.mock import patch
import pytest
from tools import fourth_goal_trial_bandit_v7 as v7

NOW=1000000.
PROBS={"rest":.6,"wander":.25,"play":.15}
def db():
 c=sqlite3.connect(":memory:")
 c.row_factory=sqlite3.Row
 c.executescript("""
 create table brain_9plus3_trial_intents(id integer primary key,
    profile text,context_key text,candidate text,at real,source_ids_json text);
 create table world_events(id integer primary key,profile text,ts real,action text);
 create table brain_9plus3_task_outcomes(id integer primary key,
   profile text,event_id integer,context_key text,action text,
   source_kind text,task_success integer);
 """)
 return c
def add(c,t,action="wander",passed=False,matching=True):
 c.execute("insert into brain_9plus3_trial_intents(profile,context_key,candidate,at,source_ids_json) values(?,?,?,?,?)",
     ("sim_fourth","scenario:test",action,t,"[]"))
 c.execute("insert into world_events(profile,ts,action) values(?,?,?)",
     ("sim_fourth",t,{"rest":"休息","wander":"探索","play":"遊び"}[action] if matching else "別の行動"))
 eid=c.execute("select last_insert_rowid()").fetchone()[0]
 c.execute("insert into brain_9plus3_task_outcomes(profile,event_id,context_key,action,source_kind,task_success) values(?,?,?,?,?,?)",
     ("sim_fourth",eid,"scenario:test",action,"simulation_environment",int(passed)))
def test_requires_fourth_identity():
 with db() as c:
  with pytest.raises(ValueError):
   v7.gate(c,"iincho_n",NOW,"scenario:test",PROBS,"rest")
def test_two_confirmed_failed_probes_hold_off_for_3days():
 with db() as c:
  add(c,NOW-3600,passed=False)
  add(c,NOW-86400,passed=False)
  with patch.object(v7.v6,"gate",side_effect=AssertionError("never reached")):
   result=v7.gate(c,"sim_fourth",NOW,"scenario:test",PROBS,"rest")
  assert result["reason"]=="verified_probe_failures_cooldown"
  assert result["trial_outcomes"]==[0,0]
  assert result["weights"]==v7.v6.original.decide_probabilities(PROBS,"rest",weight=.42)
def test_unexecuted_intent_does_not_count_as_failed_action():
 with db() as c:
  add(c,NOW-3600,passed=False,matching=False)
  add(c,NOW-86400,passed=False)
  expected={"trial_action_requested":None,"reason":"no_decline"}
  with patch.object(v7.v6,"gate",return_value=expected):
   result=v7.gate(c,"sim_fourth",NOW,"scenario:test",PROBS,"rest")
  assert result["trial_outcomes"]==[0]
  assert result["reason"]=="no_decline"
def test_successful_prior_probe_does_not_block():
 with db() as c:
  add(c,NOW-3600,passed=True)
  add(c,NOW-86400,passed=False)
  with patch.object(v7.v6,"gate",return_value={"trial_action_requested":"play","reason":"trial"}):
   result=v7.gate(c,"sim_fourth",NOW,"scenario:test",PROBS,"rest")
  assert result["trial_action_requested"]=="play"
  assert result["trial_outcomes"]==[1,0]
def test_old_failures_do_not_block_new_trials():
 with db() as c:
  add(c,NOW-86400*5,passed=False)
  add(c,NOW-86400*6,passed=False)
  with patch.object(v7.v6,"gate",return_value={"trial_action_requested":None,"reason":"allowed"}):
   assert v7.gate(c,"sim_fourth",NOW,"scenario:test",PROBS,"rest")["reason"]=="allowed"
