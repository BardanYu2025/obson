"""Read-only asset recovery audit; CPU feature arithmetic only, no model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
from obson.babel import data, representation
from obson.babel import state_rollout_data as features
from obson.babel.ae_extend import atomic_json
from obson.babel.state_coverage_data import row_audit

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = REPO/'reports/observed_history_review_cecb3cc29636/babel_observed_history768'
PEERS = [REPO/'reports'/p for p in ['multiscale_review_8b4fdb27d5c1/babel_multiscale768_v2','layer_supervision_review_c95e373e9b9d/babel_layer_supervision768']]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=json.loads((SOURCE/'manifest.json').read_text())
source_meta=meta['coverage_meta']['source_meta']
expected=source_meta['cache_source']['files']
stats=meta['statistics']
atomic_json(stats,OUT/'statistics.json')
assert sha(OUT/'statistics.json')==expected['statistics.json']
rows={s:json.loads((SOURCE/f'{s}_rows.json').read_text()) for s in ['train','val','test','cross_research']}
for split,rr in rows.items():
 assert all(json.loads((p/f'{split}_rows.json').read_text())==rr for p in PEERS)
 atomic_json(rr, OUT/f'{split}_rows.json')
print('Recovered original normalization SHA and agreed row identities from three reports',flush=True)
checks={'statistics_exact_sha':sha(OUT/'statistics.json'), 'expected_statistics_sha':expected['statistics.json'], 'row_agreement_reports':[str(p.relative_to(REPO)) for p in [SOURCE,*PEERS]], 'partition_checks':{},'splits':{},'model_executions':0,'model_weights_recovered':False}
for a,b in [('train','val'),('train','test'),('val','test')]:checks['partition_checks'][a+'/'+b]=row_audit(rows[a],rows[b])
bounds={'train_until':'2024-08-07','val_until':'2025-08-25','test_until':'2026-09-11'}
for cross,splits in [(False,['train','val','test']),(True,['cross_research'])]:
 symbols=sorted({r['symbol'] for s in splits for r in rows[s]})
 series,_=data.load_series(REPO/'data/contracts',symbols,(15,30,60),asof='2026-09-12 03:00:00')
 lookup={s.key:s for s in series}
 encoded={}
 for split in splits:
  blocks=[]
  for r in rows[split]:
   s=lookup[r['key']];end=r['row']
   assert str(s.frame.datetime.iloc[end])==r['end']
   assert str(s.frame.datetime.iloc[end-254])==r['input_start']
   same=representation.time_mask(s,bounds,'test' if cross else split)
   assert end>=254 and np.all(same[end-254:end+1])
   if s.key not in encoded:encoded[s.key]=features.encode(s.frame,s.period)
   blocks.append(((encoded[s.key][end-254:end+1]-stats['x_mean'])/stats['x_scale']).astype(np.float32))
  x=np.asarray(blocks);assert np.isfinite(x).all()
  np.savez(OUT/f'{split}.npz',x=x)
  checks['splits'][split]={'rows':len(rows[split]),'shape':list(x.shape),'cache_sha256':sha(OUT/f'{split}.npz'),'expected_cache_sha256':expected[f'cache/{split}.npz'],'cache_bytes_equal':sha(OUT/f'{split}.npz')==expected[f'cache/{split}.npz'],'row_sha256':sha(OUT/f'{split}_rows.json'),'expected_row_sha256':expected[f'cache/{split}_rows.json'],'row_bytes_equal':sha(OUT/f'{split}_rows.json')==expected[f'cache/{split}_rows.json']}
  print(split,checks['splits'][split],flush=True)
  del x, blocks
checks['input_sources']={str(p.relative_to(REPO)):sha(p) for p in [SOURCE/'manifest.json',*[SOURCE/f'{s}_rows.json' for s in rows]]}
checks['script_sha256']=sha(__file__)
atomic_json(checks,OUT/'audit.json')
