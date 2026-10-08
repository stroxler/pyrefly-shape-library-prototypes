"""Keep the semantic typing counterexamples active in the example corpus."""

import subprocess
import unittest
from pathlib import Path


class StaticTypeRegressionsTest(unittest.TestCase):
    def test_negative_probes_still_fail(self) -> None:
        """Fail if a suppressed negative probe stops producing its expected error."""
        project = Path(__file__).resolve().parents[1]
        checker = project.parent / ".venv" / "bin" / "pyrefly"
        result = subprocess.run(
            [str(checker), "check", "-c", "pyrefly.toml", "--error", "unused-ignore"],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
