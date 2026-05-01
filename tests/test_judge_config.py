from __future__ import annotations

import os
import runpy
import unittest
from unittest.mock import patch


class JudgeSimulatorConfigTests(unittest.TestCase):
    def test_reads_runtime_config_from_environment(self) -> None:
        env = {
            **os.environ,
            "BOT_URL": "https://example.test",
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-gemini-key",
            "LLM_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            module = runpy.run_path("judge_simulator.py", run_name="judge_config_test")

        self.assertEqual(module["BOT_URL"], "https://example.test")
        self.assertEqual(module["LLM_PROVIDER"], "gemini")
        self.assertEqual(module["LLM_API_KEY"], "test-gemini-key")
        self.assertEqual(module["LLM_MODEL"], "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
