"""No network calls: test the isolated model-output audit gate."""
import unittest
from experiments.fourth_llm_speech_audit import make_input, validate, evaluate

ROW = {"day": 40, "arm": "extended", "seed": 992, "scenario": "shift30",
       "focus": "activity:craft", "source_action": "小物づくりを試す",
       "sources": ["world_events#666", "world_events#665"], "evidence_count": 19,
       "speech": "代理生成テキストは入力しない"}
GOOD = {"speak": True, "speech": "小物づくりを試した記録がある",
        "reflection": "活動の頻度が高い。結果は不明。",
        "source_ids": ["world_events#666"], "focus": "activity:craft",
        "interpretation": "制作を続けている可能性",
        "uncertainty": "成功や好みは未確認", "proposed_next_action": None}

class TestFourthAudit(unittest.TestCase):
    def test_no_surrogate_leak(self):
        self.assertNotIn("speech", make_input(ROW))
    def test_valid(self):
        self.assertEqual([], validate(make_input(ROW), GOOD))
    def test_poisoned_citation(self):
        self.assertIn("unsupported_citation", validate(make_input(ROW),
            {**GOOD, "source_ids": ["world_events#999"]}))
    def test_false_silence(self):
        self.assertIn("silence_with_speech", validate(make_input(ROW),
            {**GOOD, "speak": False}))
    def test_silence_allowed(self):
        a = {**GOOD, "speak": False, "speech": "", "source_ids": []}
        self.assertEqual([], validate(make_input(ROW), a))
    def test_focus_change_rejected(self):
        self.assertIn("unsupported_focus", validate(make_input(ROW),
            {**GOOD, "focus": "operation:play"}))
    def test_semantics_not_claimed(self):
        report = evaluate([ROW], [GOOD])
        self.assertTrue(report["episodes"][0]["needs_human_semantic_review"])
        self.assertFalse(report["production_N_modified"])

if __name__ == "__main__":
    unittest.main()
