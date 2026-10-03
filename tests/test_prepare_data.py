import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import prepare, prompt_key


class PrepareDataTest(unittest.TestCase):
    def test_deduplicates_and_keeps_prompts_in_one_split(self):
        rows = [
            {"instruction": "同一问题", "input": "", "output": "回答甲"},
            {"instruction": "同一问题", "input": "", "output": "回答乙"},
            {"instruction": "同一问题", "input": "", "output": "回答甲"},
            {"instruction": "另一个问题", "input": "", "output": "回答丙"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            stats = prepare(source, root / "out")
            splits = [json.loads((root / "out" / f"{name}.json").read_text(encoding="utf-8"))
                      for name in ("train", "valid", "test")]
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        owners = [i for i, split in enumerate(splits) if any(prompt_key(x) == "同一问题" for x in split)]
        self.assertEqual(len(owners), 1)
        self.assertEqual(sum(len(x) for x in splits), 3)


if __name__ == "__main__":
    unittest.main()

