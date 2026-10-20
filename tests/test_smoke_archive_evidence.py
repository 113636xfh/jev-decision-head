"""The shipped 20-row fallback is explicitly derived, not the original live smoke."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from jev.check_baseline import BASELINES, compare

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/smoke-20-dev-20261020'


class DerivedEvidence(unittest.TestCase):
    def test_derived_rows_and_provenance_are_bound(self):
        provenance = json.loads((EVIDENCE / 'rows-smoke20-derived-provenance-v2.json').read_text())
        content = (EVIDENCE / provenance['derived_file']).read_bytes()
        rows = [json.loads(line) for line in content.splitlines()]
        self.assertEqual(hashlib.sha256(content).hexdigest(), provenance['derived_sha256'])
        self.assertFalse(provenance['original_smoke20_rows_recovered'])
        self.assertEqual(provenance['evidence_kind'], 'derived-archived-decision-replay')
        self.assertEqual(len(rows), 20)
        self.assertEqual([r['uid'] for r in rows], provenance['uids'])
        self.assertEqual(sum(r['correct_old'] for r in rows), 13)
        for row in rows:
            self.assertEqual(set(row), set(provenance['copied_fields']))
            self.assertNotIn('pred_new', row)
            self.assertNotIn('api_ms', row)
        baseline = BASELINES / 'middle5-s1.dev-predictions.jsonl'
        passed = compare(EVIDENCE / provenance['derived_file'], baseline, expected_count=20)
        self.assertTrue(passed['passed'], passed)
        self.assertEqual(passed['compared'], 20)
        self.assertFalse(compare(EVIDENCE / provenance['derived_file'], baseline)['passed'])

    def test_original_archive_projection_when_source_available(self):
        provenance = json.loads((EVIDENCE / 'rows-smoke20-derived-provenance-v2.json').read_text())
        path = Path(provenance['source_archive_path'])
        if not path.is_file():
            self.skipTest('external original archive unavailable; portable evidence hash test still runs')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance['source_archive_sha256'])
        with tarfile.open(path) as archive:
            content = archive.extractfile(provenance['source_member']).read()
        self.assertEqual(hashlib.sha256(content).hexdigest(), provenance['source_member_sha256'])
        original = [json.loads(line) for line in content.splitlines() if line.strip()]
        self.assertEqual(len(original), 1996)
        expected = [{k: row[k] for k in provenance['copied_fields']} for row in original[:20]]
        actual = [json.loads(line) for line in (EVIDENCE / provenance['derived_file']).read_text().splitlines()]
        self.assertEqual(actual, expected)
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / 'full-dev.jsonl'
            full.write_bytes(content)
            result = compare(full, BASELINES / 'middle5-s1.dev-predictions.jsonl',
                             expected_count=1996, require_complete=True)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['compared'], 1996)


if __name__ == '__main__':
    unittest.main()
