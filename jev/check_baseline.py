"""Compare archived-head decisions against the published accepted baseline, offline.

Supported row families are accepted predictions (gold/pred/probabilities), archived
replay rows (status/gold/pred_old/n_options), and release head-route predictions
(options/independent_gold/choice/head_probabilities). They are never guessed from a
single shared field. Decode-route results are not archived-head evidence.

Every supplied row must be valid, unique and present in the accepted baseline.
Full accepted coverage is the default. For intentional smoke subsets,
--expected-count binds their size; --require-complete still requires every accepted uid.
This checks recorded decisions, not feature extraction, live compatibility, or
probability equality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent / 'baselines'
PUBLISHED_BASELINE = BASELINES / 'middle5-s1.dev-predictions.jsonl'
# Published git artifact bytes, with CRLF normalized to LF for Windows checkouts.
PUBLISHED_BASELINE_SHA256_LF = '7e76aa9a6a3df0754b7e6e55cb5c1d1d690b08d896d9d249556ec25cb74b9db2'


class InvalidRows(ValueError):
    """An input is not unambiguous, comparable decision evidence."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRows('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise InvalidRows('nonfinite JSON constant: ' + value)


def _index(value, size, name):
    if type(value) is not int or not 0 <= value < size:
        raise InvalidRows('%s must be an integer in [0, %d)' % (name, size))
    return value


def _probabilities(values):
    if not isinstance(values, list) or len(values) < 2:
        raise InvalidRows('probabilities must be a list with at least two candidates')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
           for p in values):
        raise InvalidRows('probabilities must be finite numbers in [0, 1]')
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise InvalidRows('probabilities must sum to one (absolute tolerance 1e-5)')
    return max(range(len(values)), key=values.__getitem__)


def normalize(row):
    """Validate a complete comparable row; return its schema and indexed decision."""
    if not isinstance(row, dict):
        raise InvalidRows('row must be a JSON object')
    for key in ('uid', 'split'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise InvalidRows('missing or invalid ' + key)
    if row.get('error') or row.get('errors'):
        raise InvalidRows('row carries explicit error metadata')
    baseline = bool({'pred', 'probabilities'} & row.keys())
    replay = bool({'status', 'pred_old'} & row.keys())
    release = bool({'route', 'choice', 'head_probabilities', 'independent_gold', 'options'} & row.keys())
    if sum((baseline, replay, release)) != 1:
        raise InvalidRows('unsupported or ambiguous schema (baseline, replay, or head-route required)')
    schema = 'baseline' if baseline else 'replay' if replay else 'release-head'
    required = {'baseline': {'gold', 'pred', 'probabilities'},
                'replay': {'status', 'gold', 'pred_old', 'n_options'},
                'release-head': {'route', 'options', 'independent_gold', 'choice',
                                 'head_probabilities'}}[schema]
    missing = required - row.keys()
    if missing:
        raise InvalidRows('missing %s fields: %s' % (schema, ', '.join(sorted(missing))))
    # Explicit schema tags are not silently reinterpreted as a compatible family.
    if 'schema' in row:
        raise InvalidRows('unsupported explicit schema tag: %r' % row['schema'])
    if schema == 'baseline':
        if {'pred_new', 'n_options', 'options', 'head_probabilities'} & row.keys():
            raise InvalidRows('mixed baseline and replay/release prediction schema')
        best = _probabilities(row['probabilities'])
        size = len(row['probabilities'])
        gold = _index(row['gold'], size, 'gold')
        pred = _index(row['pred'], size, 'pred')
        if pred != best:
            raise InvalidRows('pred disagrees with probabilities argmax')
    elif schema == 'replay':
        if row['status'] != 'ok':
            raise InvalidRows('replay status must be ok, not %r' % row['status'])
        size = row['n_options']
        if type(size) is not int or size < 2:
            raise InvalidRows('n_options must be an integer >= 2')
        gold = _index(row['gold'], size, 'gold')
        pred = _index(row['pred_old'], size, 'pred_old')
        if 'pred_new' in row:
            _index(row['pred_new'], size, 'pred_new')
        if 'correct_old' in row and (type(row['correct_old']) not in (int, bool)
                                     or row['correct_old'] != int(pred == gold)):
            raise InvalidRows('correct_old disagrees with pred_old/gold')
    else:
        if row['route'] != 'head':
            raise InvalidRows('only head-route predictions are comparable; decode is not archived-head evidence')
        options = row['options']
        if (not isinstance(options, list) or len(options) < 2
                or any(not isinstance(x, str) or not x.strip() for x in options)
                or len(set(options)) != len(options)):
            raise InvalidRows('options must contain at least two distinct nonempty strings')
        best = _probabilities(row['head_probabilities'])
        size = len(options)
        if len(row['head_probabilities']) != size:
            raise InvalidRows('options/head_probabilities length mismatch')
        if row['independent_gold'] not in options or row['choice'] not in options:
            raise InvalidRows('independent_gold and choice must be explicit option labels')
        gold, pred = options.index(row['independent_gold']), options.index(row['choice'])
        if pred != best:
            raise InvalidRows('choice disagrees with head_probabilities argmax')
        if {'gold', 'pred_new', 'n_options'} & row.keys():
            raise InvalidRows('mixed indexed and label-based prediction schema')
    return schema, {'uid': row['uid'], 'split': row['split'], 'gold': gold,
                    'pred': pred, 'n_options': size}


def load_rows(path, accepted=False):
    rows, schemas, seen = [], set(), set()
    try:
        with Path(path).open(encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
                    schema, normalized = normalize(row)
                    if accepted and schema != 'baseline':
                        raise InvalidRows('accepted file must use the published baseline schema')
                    if normalized['uid'] in seen:
                        raise InvalidRows('duplicate uid: ' + normalized['uid'])
                    seen.add(normalized['uid'])
                    schemas.add(schema)
                    if len(schemas) > 1:
                        raise InvalidRows('mixed row schemas within one file')
                    rows.append(normalized)
                except (ValueError, TypeError, OverflowError) as exc:
                    raise InvalidRows('%s:%d: %s' % (path, number, exc)) from exc
    except (OSError, UnicodeError) as exc:
        raise InvalidRows('%s: %s' % (path, exc)) from exc
    if not rows:
        raise InvalidRows('%s: no decision rows' % path)
    return rows, next(iter(schemas))


def compare(rows_file, accepted_file, expected_count=None, require_complete=False):
    """Fail closed on malformed evidence, incomplete requested coverage, or any mismatch."""
    require_complete = require_complete or expected_count is None
    result = {'accepted_file': str(accepted_file), 'accepted_rows': 0,
              'rows_file': str(rows_file), 'rows_ok': 0, 'compared': 0,
              'gold_mismatches': 0, 'prediction_mismatches': 0,
              'unknown_uids': 0, 'incompatible_rows': 0, 'missing_accepted_rows': None,
              'expected_count': expected_count, 'require_complete': require_complete,
              'passed': False, 'errors': [], 'details': []}
    try:
        if expected_count is not None and (type(expected_count) is not int or expected_count <= 0):
            raise InvalidRows('expected_count must be a positive integer')
        published = Path(accepted_file).resolve() == PUBLISHED_BASELINE.resolve()
        result['accepted_identity'] = 'published-sha256-lf' if published else 'custom-unverified'
        if published:
            try:
                raw = Path(accepted_file).read_bytes().replace(b'\r\n', b'\n')
            except OSError as exc:
                raise InvalidRows('cannot read published baseline: %s' % exc) from exc
            result['accepted_sha256_lf'] = hashlib.sha256(raw).hexdigest()
            if result['accepted_sha256_lf'] != PUBLISHED_BASELINE_SHA256_LF:
                raise InvalidRows('published baseline digest mismatch')
        references, _ = load_rows(accepted_file, accepted=True)
        result['accepted_rows'] = len(references)
        if published and len(references) != 1996:
            raise InvalidRows('published baseline must contain exactly 1996 rows')
        accepted = {r['uid']: r for r in references}
        rows, schema = load_rows(rows_file)
        result.update(rows_ok=len(rows), rows_schema=schema)
        result['missing_accepted_rows'] = len(accepted.keys() - {r['uid'] for r in rows})
        if expected_count is not None and len(rows) != expected_count:
            result['errors'].append('expected %d rows, got %d' % (expected_count, len(rows)))
        if require_complete and result['missing_accepted_rows']:
            result['errors'].append('missing %d accepted uids' % result['missing_accepted_rows'])
        for row in rows:
            reference = accepted.get(row['uid'])
            if reference is None:
                result['unknown_uids'] += 1
                result['details'].append({'uid': row['uid'], 'why': 'unknown uid'})
                continue
            if any(reference[k] != row[k] for k in ('split', 'n_options')):
                result['incompatible_rows'] += 1
                result['details'].append({'uid': row['uid'], 'why': 'split/candidate count mismatch'})
                continue
            result['compared'] += 1
            for field, counter in [('gold', 'gold_mismatches'), ('pred', 'prediction_mismatches')]:
                if reference[field] != row[field]:
                    result[counter] += 1
                    result['details'].append({'uid': row['uid'],
                                              'why': 'gold' if field == 'gold' else 'pred_archived',
                                              'accepted': reference[field], 'mine': row[field]})
        result['passed'] = (not result['errors'] and result['compared'] == len(rows)
                            and not any(result[k] for k in ('gold_mismatches', 'prediction_mismatches',
                                                          'unknown_uids', 'incompatible_rows')))
    except InvalidRows as exc:
        result['errors'].append(str(exc))
    result['details'] = result['details'][:20]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', required=True, help='baseline, archived replay, or release head JSONL')
    parser.add_argument('--accepted', default=str(PUBLISHED_BASELINE),
                        help='published baseline (digest checked); custom references are marked unverified')
    parser.add_argument('--out', default='')
    parser.add_argument('--expected-count', type=int,
                        help='require exactly this many unique valid rows; permits an explicit smoke subset')
    parser.add_argument('--require-complete', action='store_true',
                        help='require all accepted uids even with --expected-count (default without it)')
    args = parser.parse_args()
    result = compare(args.rows, args.accepted, args.expected_count, args.require_complete)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    print(text)
    if args.out:
        # A check must not silently replace older evidence.
        with Path(args.out).open('x', encoding='utf-8') as handle:
            handle.write(text)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
