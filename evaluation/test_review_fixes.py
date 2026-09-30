"""Small regression checks for the artifact evaluation wrappers."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import common


class ReproductionModeTests(unittest.TestCase):
    def test_local_tables_keep_only_available_models(self):
        methods = [("DA3", "da3"), ("GRADE", "ours_full")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ours_full.csv").touch()
            with patch.object(common, "MERGED_DIR", root), patch.dict(
                os.environ, {"GRADE_REPRODUCTION_MODE": "local"}
            ):
                self.assertEqual(common.available_methods(methods), methods[1:])
                (root / "ours_full.csv").unlink()
                with self.assertRaises(FileNotFoundError):
                    common.available_methods(methods)
            with patch.dict(os.environ, {"GRADE_REPRODUCTION_MODE": "saved"}):
                self.assertEqual(common.available_methods(methods), methods)


if __name__ == "__main__":
    unittest.main()
