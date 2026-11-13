"""Check long-task split selection without BFCL, a model, or Raspberry Pi."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from harness.task_split import TaskSplitError, load_task_ids, select_task_ids

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits" / "tasks_TradingBot_MessageAPI.json"
LONG_TASKS = ROOT / "data" / "long_tasks.jsonl"


class LongTaskSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = json.loads(SPLIT.read_text(encoding="utf-8"))
        cls.all_long = [f"a1_long_context_{n}" for n in range(1, 7)]

    def test_original_bfcl_split_unchanged(self):
        self.assertEqual(len(select_task_ids(self.split, "test")), 13)
        self.assertEqual(len(select_task_ids(self.split, "trace")), 20)

    def test_split_custom_long_tasks(self):
        test = select_task_ids(self.split, "test", self.all_long)
        trace = select_task_ids(self.split, "trace", self.all_long)
        self.assertEqual(len(test), 18)
        self.assertEqual(len(trace), 21)
        self.assertIn("a1_long_context_4", trace)
        self.assertNotIn("a1_long_context_4", test)
        self.assertEqual(set(test) & set(trace), set())
        self.assertEqual(len(select_task_ids(self.split, "all", self.all_long)), 39)
        self.assertEqual(select_task_ids(self.split, "dev", self.all_long), trace)

    def test_all_six_custom_tasks_in_manifest(self):
        self.assertEqual(
            set(self.split["extra_tasks"]["test"] + self.split["extra_tasks"]["trace"]),
            set(self.all_long),
        )

    def test_actual_jsonl_ids(self):
        self.assertEqual(set(load_task_ids(LONG_TASKS)), set(self.all_long))
        rows = [json.loads(line) for line in LONG_TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
        for task in rows:
            self.assertEqual(len(task["question"]), len(task["ground_truth"]))
            self.assertTrue(task["involved_classes"])

    def test_missing_assignment_fails_closed(self):
        changed = json.loads(json.dumps(self.split))
        changed["extra_tasks"]["test"].remove("a1_long_context_1")
        with self.assertRaisesRegex(TaskSplitError, "unassigned"):
            select_task_ids(changed, "test", self.all_long)

    def test_extra_tasks_without_manifest_fails(self):
        changed = dict(self.split)
        changed.pop("extra_tasks")
        with self.assertRaisesRegex(TaskSplitError, "requires extra_tasks"):
            select_task_ids(changed, "test", self.all_long)

    def test_cross_split_leak_fails(self):
        changed = json.loads(json.dumps(self.split))
        changed["extra_tasks"]["trace"].append("a1_long_context_1")
        with self.assertRaisesRegex(TaskSplitError, "both test and trace"):
            select_task_ids(changed, "test", self.all_long)

    def test_unknown_custom_task_fails(self):
        with self.assertRaisesRegex(TaskSplitError, "not_loaded"):
            select_task_ids(self.split, "test", self.all_long[:-1])

    def test_custom_id_collision_with_bfcl_fails(self):
        changed = json.loads(json.dumps(self.split))
        changed["extra_tasks"]["test"][0] = changed["test"][0]
        supplied = [changed["test"][0]] + self.all_long[1:]
        with self.assertRaisesRegex(TaskSplitError, "overlap BFCL"):
            select_task_ids(changed, "test", supplied)

    def test_bad_split_fails(self):
        with self.assertRaises(TaskSplitError):
            select_task_ids(self.split, "invalid", self.all_long)

    def test_duplicate_jsonl_ids_fail(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "duplicates.jsonl"
            path.write_text('{"id":"x"}\n{"id":"x"}\n', encoding="utf-8")
            with self.assertRaisesRegex(TaskSplitError, "duplicate"):
                load_task_ids(path)


if __name__ == "__main__":
    unittest.main()
