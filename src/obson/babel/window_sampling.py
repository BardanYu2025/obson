"""Immutable train-only candidate cache and reproducible, budget-matched endpoint sampling."""
from collections import Counter
from pathlib import Path

import numpy as np

from . import architecture as ar, architecture_benchmark as ab, capacity_benchmark as cb
from . import coverage_audit as cov, data, ae_context, activity_ablation as aa, pca_teacher as pt
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SAMPLING_VERSION = 'uniform-without-replacement-per-epoch-v1'


def coverage_identity(root, source):
    done = read_json(root/'completion.json')
    if done['status'] != 'complete' or done['training_updates'] or done['cache_mutations']:
        raise ValueError('Expected completed, read-only coverage audit')
    verify_files(root, done['files'])
    metrics = read_json(root/'coverage_metrics.json')
    if metrics['schema'] != cov.SCHEMA or metrics['reports_only'] or not metrics['all_cached_arrays_verified']:
        raise ValueError('Strict cache coverage audit required')
    if Path(metrics['sources']['architecture']).resolve() != source.resolve():
        raise ValueError('Coverage and teacher architecture sources differ')
    for name, digest in metrics['code_sha256'].items():
        cov.file_check(Path(cov.__file__).parent/name, digest)
    for name, digest in metrics['consumed_report_sha256'].items():
        cov.file_check(Path(name), digest)
    index = read_json(source/'cache/index.json')['files']
    for split, rows in metrics['cache_replay'].items():
        if read_json(root/f'{split}_cache_replay.json') != rows:
            raise ValueError('Coverage replay summary differs')
        for name, row in rows.items():
            if not row['source_binary_available'] or not row['numeric_replay_passed'] or row['expected_sha256'] != index[f'{split}_{name}.npy']:
                raise ValueError('Unverified or mismatched original arrays')
    if set(metrics['cache_replay']) != {'train','val','test','cross_research'} or any(set(r) != {'x','y','mask'} for r in metrics['cache_replay'].values()):
        raise ValueError('Incomplete coverage replay')
    return dict(files=done['files'] | {'completion.json': sha256(root/'completion.json')},
                train_count=metrics['profiles']['train']['coverage']['windows'],
                candidate=metrics['train_candidates']['history512_stride16'],
                boundaries=metrics['boundaries'], sources=metrics['sources'])


def sample_ids(count, budget, seed, epoch):
    if not 0 < budget <= count or epoch < 1:
        raise ValueError('Invalid per-epoch sampling budget or epoch')
    # Membership RNG is independent of torch/global RNG and the existing minibatch shuffle.
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20260923]))
    return np.sort(rng.choice(count, budget, replace=False)).astype(np.int64)


def unique_bars(keys, ends, ids):
    if not len(ids):
        return 0
    k, e = keys[ids], ends[ids]
    order = np.lexsort((e, k)); k, e = k[order], e[order]
    return int(128 + np.where(k[1:] == k[:-1], np.minimum(128, np.diff(e)), 128).sum())


def sampling_plan(inventory, seed, epochs, budget):
    n = len(inventory)
    keys = np.unique([r['key'] for r in inventory], return_inverse=True)[1]
    ends = np.array([r['row'] for r in inventory])
    strata = {f: np.unique([str(r[f]) for r in inventory], return_inverse=True)
              for f in ('symbol', 'period', 'month')}
    counts = np.zeros(n, dtype=np.int64); rows = []
    for epoch in range(1, epochs+1):
        ids = sample_ids(n, budget, seed, epoch); counts[ids] += 1
        groups = {f: {str(k): int(v) for k, v in zip(labels, np.bincount(codes[ids], minlength=len(labels)))}
                  for f, (labels, codes) in strata.items()}
        seen = np.flatnonzero(counts)
        rows.append(dict(epoch=epoch, ids_sha256=cov.ndarray_hash(ids), windows=len(ids), strata=groups,
                         unique_input_bars=unique_bars(keys, ends, ids), cumulative_unique_windows=len(seen),
                         cumulative_unique_input_bars=unique_bars(keys, ends, seen)))
    return dict(version=SAMPLING_VERSION, seed=seed, budget=budget, pool_size=n, epochs=rows,
                total_window_exposures=int(counts.sum()), exposure_counts=counts.tolist(),
                cumulative_strata={f: {str(k): int(v) for k, v in zip(labels, np.bincount(codes, weights=counts, minlength=len(labels)))}
                                   for f, (labels, codes) in strata.items()})


class SubsetArray:
    """Index a read-only mmap only when a minibatch is requested; never copy the full pool."""
    def __init__(self, array, ids):
        self.array, self.ids = array, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.array[self.ids[index]]


def subset(data_arrays, ids):
    return {k: SubsetArray(v, ids) for k, v in data_arrays.items()}


def load_candidates(out):
    return {k: np.load(out/f'candidates/train_{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('x','y','mask','q')}


def verify_cache(meta, out):
    cache = out/'candidates'; index = read_json(cache/'index.json')
    if index['manifest'] != meta:
        raise ValueError('Candidate cache identity mismatch')
    verify_files(cache, index['files'])
    verify_files(out/'coverage', meta['coverage_identity']['files'])


def write_candidates(series, encoded, keys, stats, pca, scales, old_inventory, old_data, cache):
    """Chunked target construction; every retained legacy window must numerically replay."""
    old = {(r['key'], r['row']): i for i, r in enumerate(old_inventory)}
    if len(old) != len(old_inventory):
        raise ValueError('Duplicate original endpoint')
    n = len(keys); arrays = {}; inventory = []; matched = set(); maxdiff = dict(x=0., y=0., mask=0.)
    for left in range(0, n, 256):
        block = keys[left:left+256]
        x = np.stack([encoded[i][end-127:end+1] for i, end in block])
        y, mask = ar.ordered_targets(x); x, y, mask = ar.normalize(x, y, mask, stats)
        values = dict(x=x, y=y, mask=mask)
        values['q'] = pt.normalize_coordinates(pt.coefficients(pca, values), scales)
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite candidate array')
            if name not in arrays:
                arrays[name] = np.lib.format.open_memmap(cache/f'train_{name}.npy', mode='w+', dtype=value.dtype, shape=(n,)+value.shape[1:])
            arrays[name][left:left+len(block)] = value
        for j, (i, end) in enumerate(block):
            s = series[i]; dt = s.frame.datetime.iloc[end]
            row = dict(key=s.key, symbol=s.code, period=s.period, row=end, end=str(dt), month=str(dt.to_period('M')))
            inventory.append(row)
            identity = (s.key, end)
            if identity in old:
                idx = old[identity]; matched.add(identity)
                if str(dt) != old_inventory[idx]['end']:
                    raise ValueError('Original endpoint timestamp changed')
                for name in maxdiff:
                    a, b = values[name][j], old_data[name][idx]
                    delta = float(np.abs(a.astype(float)-b.astype(float)).max()); maxdiff[name] = max(maxdiff[name], delta)
                    if (name == 'mask' and not np.array_equal(a,b)) or not np.allclose(a,b,atol=1e-6,rtol=2e-5):
                        raise ValueError(f'Candidate does not reproduce original {identity}/{name}: {delta}')
        if left % 4096 == 0:
            progress(f'Candidate cache: {min(left+256,n)}/{n} train windows, no model updates')
    if matched != set(old):
        raise ValueError('Dense pool lost original training endpoints')
    for array in arrays.values():
        array.flush()
    return inventory, dict(original_windows_matched=len(matched), max_abs=maxdiff, atol=1e-6, rtol=2e-5)


def prepare_candidates(meta, out):
    cache = out/'candidates'; cache.mkdir(exist_ok=True)
    if (cache/'index.json').exists():
        verify_cache(meta, out); return
    source = Path(meta['source']); identity = meta['coverage_identity']
    ref = read_json(Path(identity['sources']['long'])/'manifest.json')['manifest']
    keys = [r['key'].split('/') for r in ref['sources']]
    series, _ = data.load_series(Path(meta['raw_root']), sorted({r[0] for r in keys}), sorted({int(r[1]) for r in keys}))
    if data.manifest(series, identity['boundaries']) != ref:
        raise ValueError('Raw candidate source differs from audited training source')
    # Features may read strictly preceding context; only train endpoints can become examples.
    encoded = [np.column_stack((ae_context.encode_context(s.frame,s.period,'ema8_32')['x'], aa.activity_features(s.frame,s.period)[0])) for s in series]
    candidates = [(i, int(e)) for i, s in enumerate(series)
                  for e in cov.endpoints(s, identity['boundaries'], 'train', stride=16, minimum=512)]
    if len(candidates) != identity['candidate']['coverage']['windows']:
        raise ValueError('Audited candidate count changed')
    stats = read_json(out/'statistics.json'); scales = read_json(out/'cache/coordinate_scales.json')
    inventory, replay = write_candidates(series, encoded, candidates, stats, cb.source_pca(meta), scales,
        read_json(out/'coverage/train_windows.json'), ab.load_arrays(source,'train'), cache)
    del encoded, series
    keys = np.unique([r['key'] for r in inventory], return_inverse=True)[1]
    ends = np.array([r['row'] for r in inventory]); bars = unique_bars(keys, ends, np.arange(len(inventory)))
    expected = identity['candidate']
    counts = Counter(r['key'] for r in inventory)
    if bars != expected['coverage']['unique_contract_period_bars'] or dict(counts) != {k:v for k,v in expected['counts_by_series'].items() if v}:
        raise ValueError('Audited candidate coverage changed')
    atomic_json(inventory, cache/'inventory.json')
    for job in meta['experiments']:
        plan = sampling_plan(inventory, job['seed'], job['epochs'], meta['windows_per_epoch'])
        atomic_json(plan, cache/f'sampling_s{job["seed"]}.json')
    atomic_json(dict(train_only=True, fitted_new_statistics=False, fitted_new_pca=False,
        original_replay=replay, pool_windows=len(inventory), unique_contract_period_bars=bars,
        current_bar_scored=False, sampling=SAMPLING_VERSION, source_boundary=identity['boundaries']), cache/'data_audit.json')
    files = {p.name: sha256(p) for p in cache.iterdir() if p.is_file() and p.name!='index.json'}
    atomic_json(dict(manifest=meta, files=files), cache/'index.json')
    verify_cache(meta, out)
