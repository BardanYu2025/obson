"""Conservative, score-blind provenance and raw quality audit for new symbols."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_frame
from .dual_state import sha256
from .progress import progress

CANDIDATES = ('RM', 'au', 'c', 'cs', 'ru')
PATH_KEYS = {'source', 'parent_run', 'activity_run', 'long_run', 'short_run', 'fusion_run',
             'encoder_run', 'fixed_reference_run'}


def read_json(path):
    return json.loads(Path(path).read_text())


def walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def records(value):
    return [r for k, v in walk(value) if k == 'sources' and isinstance(v, list)
            for r in v if isinstance(r, dict) and {'key', 'sha256', 'start', 'end'} <= r.keys()]


def manifest_links(value):
    for key, v in walk(value):
        if isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v):
            if key.endswith(('manifest', 'manifest_sha256', 'manifest.json')):
                yield v


def scan_lineage(registry, start, exclude):
    """Follow explicit path and manifest-hash edges; unresolved edges fail closed.

    Also union source lists of unrelated registered research runs, conservatively
    excluding symbols already researched. External/deleted experiments are unknown.
    """
    registry, start, exclude = map(lambda p: Path(p).resolve(), (registry, start, exclude))
    manifests = {}; by_hash = {}; issues = []; prior_holdout_symbols = set()
    for path in sorted(registry.rglob('manifest.json')):
        path = path.resolve()
        if exclude == path.parent or exclude in path.parents:
            continue
        try:
            obj = read_json(path); digest = sha256(path)
            manifests[str(path)] = obj; by_hash.setdefault(digest, []).append(str(path))
            if obj.get('schema') == 'babel-state-holdout-v1' and (path.parent/'evaluation_lock.json').exists():
                prior_holdout_symbols.update(read_json(path.parent/'raw_audit.json')['eligible_symbols'])
        except (OSError, ValueError, KeyError, TypeError) as e:
            issues.append(dict(path=str(path), reason='unreadable_registered_manifest', detail=str(e)))
    queue = [str(start/'manifest.json')]; visited = set(); roots = []; graph = {}
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        if path not in manifests:
            issues.append(dict(path=path, reason='ancestor_not_in_registry')); continue
        obj = manifests[path]; edges = []
        for k, v in obj.items():
            if k in PATH_KEYS and isinstance(v, str):
                target = Path(v)
                if not target.is_absolute():
                    issues.append(dict(path=path, reason='ambiguous_relative_ancestor', value=v)); continue
                edges.append(str((target if target.name == 'manifest.json' else target/'manifest.json').resolve()))
        for digest in set(manifest_links(obj)):
            matches = by_hash.get(digest, [])
            if matches:
                edges.append(matches[0])
            else:
                issues.append(dict(path=path, reason='unresolved_manifest_hash', sha256=digest))
        if records(obj):
            roots.append(path)
        if not edges and not records(obj):
            issues.append(dict(path=path, reason='opaque_ancestry_without_data_sources'))
        graph[path] = edges
        queue.extend(edges)
    def reaches_source(path, seen):
        if path in roots:return True
        if path in seen:return False
        return any(reaches_source(edge, seen | {path}) for edge in graph.get(path, []))
    for path in sorted(visited):
        if not reaches_source(path, set()):
            issues.append(dict(path=path, reason='ancestor_has_no_resolved_dataset_path'))
    source_records = [r for obj in manifests.values() for r in records(obj)]
    if not roots or not source_records:
        issues.append(dict(reason='no_dataset_source_manifest'))
    return dict(registry=str(registry), manifests={p:sha256(p) for p in manifests},
        ancestors=sorted(visited), dataset_roots=sorted(roots), issues=issues,
        source_records=source_records, previous_holdout_symbols=sorted(prior_holdout_symbols), verified=not issues,
        scope='Available registry and reachable declared ancestors only; deleted/external or unrecorded human research cannot be certified.')


def identity_rows(df):
    # Timestamp and OHLCV detect renamed/re-exported copies despite changed OI columns.
    return pd.util.hash_pandas_object(df[['datetime', 'open', 'high', 'low', 'close', 'volume']], index=False).to_numpy()


def strict_frame(path, period):
    df = validate_frame(pd.read_csv(path), str(path))
    if len(df) < 2 or (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Too few bars or nonpositive price')
    dt = df.datetime.diff().dt.total_seconds().to_numpy()[1:]
    if (dt < period*60).any():
        raise ValueError('Sub-period or overlapping bars')
    return df


def audit_raw(root, lineage, symbols=CANDIDATES):
    root = Path(root).resolve(); used = lineage['source_records']; known_symbols = {r['key'].split('/')[0].lower() for r in used}
    known_symbols.update(s.lower() for s in lineage.get('previous_holdout_symbols', []))
    known_contracts = {r['key'].split('/')[-1].lower() for r in used}
    known_hashes = {r['sha256'] for r in used}; row_hashes = set(); missing_sources = []; source_raw_hashes = {}
    by_key = {}
    for r in used:
        by_key.setdefault(r['key'], {})[r['sha256']] = r
    # Read all recorded symbol files, including periods not scored in this run.
    for symbol in sorted({r['key'].split('/')[0] for r in used}):
        paths = sorted((root/symbol).glob('*.csv'))
        if not paths:
            missing_sources.append(symbol)
        for path in paths:
            try:
                period = int(path.stem.rsplit('_', 1)[1].removesuffix('m'))
                frame = strict_frame(path, period)
                source_raw_hashes[str(path.resolve())] = sha256(path)
                contract = path.stem.rsplit('_', 1)[0]
                for record in by_key.get(f'{symbol}/{period}/{contract}', {}).values():
                    prefix = frame[(frame.datetime >= pd.Timestamp(record['start'])) & (frame.datetime <= pd.Timestamp(record['end']))]
                    digest = hashlib.sha256(pd.util.hash_pandas_object(prefix,index=False).values.tobytes()).hexdigest()
                    if digest != record['sha256']:
                        missing_sources.append(f'{path}: registered source content mismatch')
                row_hashes.update(identity_rows(frame).tolist())
            except (ValueError, KeyError, OSError) as e:
                missing_sources.append(f'{path}: {e}')
        progress(f'provenance: duplicate-content inventory {symbol}, {len(row_hashes):,} unique rows')
    # Missing individual registered contracts also prevent an overlap certificate.
    for key in sorted({r['key'] for r in used}):
        symbol, period, contract = key.split('/')
        if not (root/symbol/f'{contract}_{period}m.csv').exists():
            missing_sources.append(key)
    files = []; symbol_status = {}; candidate_hashes = {}; candidate_rows = {}; duplicates = set()
    for symbol in symbols:
        paths = sorted((root/symbol).glob('*.csv')); reasons = []
        if symbol.lower() in known_symbols:
            reasons.append('symbol_already_registered')
        if not paths:
            reasons.append('no_files')
        for path in paths:
            try:
                contract, suffix = path.stem.rsplit('_', 1); period = int(suffix.removesuffix('m'))
                if period not in (15, 30, 60):
                    continue
                row = dict(path=str(path), symbol=symbol, period=period, contract=contract, sha256=sha256(path))
                candidate_hashes[str(path)] = row['sha256']
                df = strict_frame(path, period)
                source_hash = hashlib.sha256(pd.util.hash_pandas_object(df,index=False).values.tobytes()).hexdigest()
                row_ids=set(identity_rows(df).tolist())
                overlap = sum(int(h) in row_hashes for h in identity_rows(df))
                for prior_symbol, ids in candidate_rows.items():
                    if prior_symbol != symbol and len(row_ids & ids) >= 128:
                        duplicates.update((symbol, prior_symbol))
                candidate_rows.setdefault(symbol,set()).update(row_ids)
                row.update(rows=len(df), start=str(df.datetime.iloc[0]), end=str(df.datetime.iloc[-1]),
                    source_hash=source_hash, matching_registered_rows=overlap, quality='valid')
                if contract.lower() in known_contracts or source_hash in known_hashes or overlap >= 128:
                    reasons.append('registered_contract_or_content_overlap')
            except (ValueError, OSError, KeyError) as e:
                row = dict(path=str(path),symbol=symbol,quality='invalid',reason=str(e))
                reasons.append('invalid_candidate_file')
            files.append(row)
        if not any(r.get('symbol')==symbol and r.get('quality')=='valid' for r in files):
            reasons.append('no_supported_period_files')
        if not lineage['verified'] or missing_sources:
            reasons.append('provenance_unresolved')
        symbol_status[symbol] = dict(eligible=not reasons,reasons=sorted(set(reasons)))
    for symbol in duplicates:
        symbol_status[symbol]['eligible']=False
        symbol_status[symbol]['reasons'].append('content_overlap_with_another_candidate_symbol')
    eligible = [s for s in symbols if symbol_status[s]['eligible']]
    return dict(symbols=symbol_status, eligible_symbols=eligible, files=files,
        raw_hashes=candidate_hashes, source_raw_hashes=source_raw_hashes, missing_source_inventory=sorted(set(missing_sources)),
        lineage_verified=lineage['verified'],
        scope='Cross-symbol exclusion within audited registry, never a global certificate. Entire symbol excluded on any quality/identity failure; no score-based filtering.')
