import unittest
from time_entry_resolution import resolve_time_entries, UnresolvedTaskError

class ResolutionTests(unittest.TestCase):
    def test_bulk_zero_recovers_task_without_changing_time(self):
        raw = [{"id": "entry", "task": "0", "duration": "3600000", "start": "123"}]
        result, count = resolve_time_entries(raw, lambda _: {"data": {
            "task": {"id": "real-task", "name": "Real task"}, "duration": "999", "start": "999"}})
        self.assertEqual(count, 1)
        self.assertEqual(result[0]["task"]["id"], "real-task")
        self.assertEqual(result[0]["duration"], "3600000")
        self.assertEqual(result[0]["start"], "123")
        self.assertEqual(raw[0]["task"], "0")

    def test_existing_task_does_not_need_lookup(self):
        raw = [{"task": {"id": "task"}, "duration": "1000"}]
        self.assertEqual(resolve_time_entries(raw, lambda _: self.fail()), (raw, 0))

    def test_unresolved_task_blocks_report(self):
        for response in ({"data": {"task": "0"}}, {"data": None}, {}):
            with self.assertRaises(UnresolvedTaskError):
                resolve_time_entries([{"id": "entry", "task": "0", "duration": "1000"}], lambda _: response)

    def test_running_timer_is_excluded(self):
        self.assertEqual(resolve_time_entries([{"duration": "-1000"}], lambda _: self.fail()), ([], 0))

if __name__ == "__main__":
    unittest.main()
