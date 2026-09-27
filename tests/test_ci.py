"""The GitHub Actions workflow matches the local unittest command."""

from __future__ import annotations

import unittest
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
_TEST_COMMAND = "python -m unittest discover -s tests -v"


class WorkflowTests(unittest.TestCase):
    def test_workflow_is_push_and_pr_on_main_with_python_312(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertTrue(_WORKFLOW.is_file())
        self.assertIn("push:", text)
        self.assertIn("pull_request:", text)
        self.assertIn("branches: [main]", text)
        self.assertIn('python-version: "3.12"', text)
        self.assertEqual(text.count(_TEST_COMMAND), 1)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("OPENROUTER", text)
        self.assertNotIn("yf.download", text)


if __name__ == "__main__":
    unittest.main()
