"""Offline contract tests, including the actual 1,996-row release baseline."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from jev.check_baseline import BASELINES, compare


class BaselineContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.accepted = self.root / 'accepted.jsonl'
        self.rows = self.root / 'rows.jsonl'
        self.baseline = {'uid': 'one', 'split': 'dev', 'gold': 1, 'pred': 1,
                         'probabilities': [0.2, 0.8]}
        self.replay = {'uid': 'one', 'split': 'dev', 'status': 'ok', 'gold': 1,
                       'pred_old': 1, 'n_options': 2}
        self.release = {'uid': 'one', 'split': 'dev', 'route': 'head',
                        'options': ['a', 'b'], 'independent_gold': 'b',
                        'choice': 'b', 'head_probabilities': [0.2, 0.8]}
        self.write(self.accepted, [self.baseline])

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    def check(self, rows, **kwargs):
        self.write(self.rows, rows)
        return compare(self.rows, self.accepted, **kwargs)

    def test_actual_baseline_self_comparison_all_1996(self):
        path = BASELINES / 'middle5-s1.dev-predictions.jsonl'
        result = compare(path, path, expected_count=1996, require_complete=True)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['compared'], 1996)
        self.assertEqual(result['accepted_identity'], 'published-sha256-lf')

    def test_published_reference_identity_is_pinned(self):
        # Redirect only the path to a disposable fixture; never mutate the real baseline.
        self.write(self.rows, [self.baseline])
        with patch('jev.check_baseline.PUBLISHED_BASELINE', self.accepted):
            result = compare(self.rows, self.accepted)
        self.assertFalse(result['passed'])
        self.assertIn('digest mismatch', result['errors'][0])

    def test_custom_reference_is_explicitly_unverified(self):
        result = self.check([self.replay])
        self.assertTrue(result['passed'])
        self.assertEqual(result['accepted_identity'], 'custom-unverified')

    def test_explicit_error_metadata_is_rejected(self):
        for row in [self.baseline, self.replay, self.release]:
            for change in [{'error': 'extraction failed'}, {'errors': ['failed']}]:
                with self.subTest(row=row, change=change):
                    self.assertFalse(self.check([dict(row, **change)])['passed'])

    def test_each_supported_schema(self):
        for schema, row in [('baseline', self.baseline), ('replay', self.replay),
                            ('release-head', self.release)]:
            with self.subTest(schema=schema):
                result = self.check([row], expected_count=1, require_complete=True)
                self.assertTrue(result['passed'], result)
                self.assertEqual(result['rows_schema'], schema)

    def test_every_required_field_is_required(self):
        for row in [self.baseline, self.replay, self.release]:
            for key in row:
                changed = copy.deepcopy(row)
                del changed[key]
                with self.subTest(schema=list(row), key=key):
                    self.assertFalse(self.check([changed])['passed'])

    def test_unknown_uid_is_not_silently_skipped(self):
        result = self.check([self.replay, dict(self.replay, uid='unknown')])
        self.assertFalse(result['passed'])
        self.assertEqual(result['unknown_uids'], 1)
        self.assertEqual(result['compared'], 1)

    def test_all_unknown_uids_fail(self):
        self.assertFalse(self.check([dict(self.replay, uid='unknown')])['passed'])

    def test_duplicate_candidate_uid_fails(self):
        self.assertFalse(self.check([self.replay, self.replay])['passed'])

    def test_duplicate_accepted_uid_fails(self):
        self.write(self.accepted, [self.baseline, self.baseline])
        self.assertFalse(self.check([self.replay])['passed'])

    def test_no_silent_error_row_filter(self):
        self.assertFalse(self.check([self.replay, dict(self.replay, uid='two', status='error')])['passed'])

    def test_replay_status_values_fail(self):
        for status in ['error', 'OK', '', None, True]:
            with self.subTest(status=status):
                self.assertFalse(self.check([dict(self.replay, status=status)])['passed'])

    def test_gold_and_prediction_mismatches_fail(self):
        for change, counter in [({'gold': 0}, 'gold_mismatches'),
                                ({'pred_old': 0}, 'prediction_mismatches')]:
            result = self.check([dict(self.replay, **change)])
            self.assertFalse(result['passed'])
            self.assertEqual(result[counter], 1)

    def test_split_and_geometry_must_match(self):
        for change in [{'split': 'test'}, {'n_options': 3}]:
            result = self.check([dict(self.replay, **change)])
            self.assertFalse(result['passed'])
            self.assertEqual(result['incompatible_rows'], 1)

    def test_replay_numeric_types_and_ranges(self):
        for key in ['gold', 'pred_old', 'n_options']:
            for value in [True, None, -1, 1.0, '1']:
                with self.subTest(key=key, value=value):
                    self.assertFalse(self.check([dict(self.replay, **{key: value})])['passed'])
        self.assertFalse(self.check([dict(self.replay, pred_old=2)])['passed'])
        self.assertFalse(self.check([dict(self.replay, pred_new=2)])['passed'])
        self.assertFalse(self.check([dict(self.replay, correct_old=0)])['passed'])

    def test_probabilities_must_be_real_distributions(self):
        for values in [[], [1], [0.2, 0.2], [float('nan'), 1],
                       [float('inf'), 0], [-0.1, 1.1], [True, 0], ['0.2', 0.8]]:
            with self.subTest(values=values):
                self.assertFalse(self.check([dict(self.baseline, probabilities=values)])['passed'])

    def test_probabilities_and_choice_must_agree(self):
        self.assertFalse(self.check([dict(self.baseline, pred=0)])['passed'])
        self.assertFalse(self.check([dict(self.release, choice='a')])['passed'])

    def test_decode_route_not_archived_head_evidence(self):
        self.assertFalse(self.check([dict(self.release, route='decode')])['passed'])

    def test_release_labels_and_geometry(self):
        for change in [{'independent_gold': None}, {'independent_gold': 'c'}, {'choice': 'c'},
                       {'options': ['a', 'a']}, {'options': ['a', 'b', 'c']},
                       {'head_probabilities': [0.1, 0.2, 0.7]}, {'gold': 1}]:
            with self.subTest(change=change):
                self.assertFalse(self.check([dict(self.release, **change)])['passed'])

    def test_schema_ambiguity_and_tags(self):
        for row in [dict(self.replay, pred=1), dict(self.baseline, status='ok'),
                    dict(self.release, pred_old=1), dict(self.baseline, pred_new=1),
                    dict(self.replay, schema='future-v99'),
                    {'uid': 'one', 'split': 'dev', 'gold': 1}]:
            with self.subTest(row=row):
                self.assertFalse(self.check([row])['passed'])

    def test_no_mixed_schema_file(self):
        self.write(self.accepted, [self.baseline, dict(self.baseline, uid='two')])
        self.assertFalse(self.check([self.baseline, dict(self.replay, uid='two')])['passed'])

    def test_accepted_must_use_baseline_schema(self):
        self.write(self.accepted, [self.replay])
        self.assertFalse(self.check([self.replay])['passed'])

    def test_subset_is_explicit_and_complete_gate_enforced(self):
        self.write(self.accepted, [self.baseline, dict(self.baseline, uid='two')])
        result = self.check([self.replay], expected_count=1)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['missing_accepted_rows'], 1)
        self.assertFalse(self.check([self.replay])['passed'])
        self.assertFalse(self.check([self.replay], require_complete=True)['passed'])
        self.assertFalse(self.check([self.replay], expected_count=2)['passed'])
        self.assertFalse(self.check([self.replay], expected_count=0)['passed'])

    def test_invalid_input_fails_closed(self):
        for text in ['', '\n', '{broken}\n', '[]\n', 'null\n',
                     '{"uid":"one","uid":"two"}\n']:
            self.rows.write_text(text, encoding='utf-8')
            with self.subTest(text=text):
                self.assertFalse(compare(self.rows, self.accepted)['passed'])
        self.rows.write_bytes(b'\xff\xfe\xff')
        self.assertFalse(compare(self.rows, self.accepted)['passed'])
        self.assertFalse(compare(self.root / 'missing.jsonl', self.accepted)['passed'])

    def test_cli_json_failure_and_success_exit_codes(self):
        for row, expected in [(self.replay, 0), (dict(self.replay, uid='unknown'), 1)]:
            self.write(self.rows, [row])
            proc = subprocess.run([sys.executable, '-m', 'jev.check_baseline',
                                   '--rows', str(self.rows), '--accepted', str(self.accepted)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, expected, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['passed'], expected == 0)

    def test_output_evidence_is_not_overwritten(self):
        self.write(self.rows, [self.replay])
        out = self.root / 'check.json'
        out.write_text('old evidence', encoding='utf-8')
        proc = subprocess.run([sys.executable, '-m', 'jev.check_baseline', '--rows', str(self.rows),
                               '--accepted', str(self.accepted), '--out', str(out)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(out.read_text(), 'old evidence')


if __name__ == '__main__':
    unittest.main()
