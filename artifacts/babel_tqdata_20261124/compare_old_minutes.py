from pathlib import Path
import json
import numpy as np
import pandas as pd
root=Path('/Users/bardan/private/tqdata_2015');old=Path('data/contracts')
inv=json.load(open('reports/tqdata_audit_20261124_final/inventory.json'))
fields=['open','high','low','close','volume','open_oi','close_oi']
rows=[]
for product in sorted(p.name for p in old.iterdir() if p.is_dir()):
 newproduct='SR' if product=='sr' else product
 candidates=[]
 for r in inv:
  if r['family']!='month' or r['product']!=newproduct or r['first']>'2026-06-01' or r['last']<'2026-06-01':continue
  contract=r['file_contract_code'].replace('_','.',1)
  if (old/product/(contract+'_15m.csv')).exists():candidates.append(r)
 if not candidates:continue
 r=sorted(candidates,key=lambda r:r['path'])[0]
 x=pd.read_parquet(root/r['path']);x['datetime']=pd.to_datetime(x.datetime);x=x.set_index('datetime')
 contract=r['file_contract_code'].replace('_','.',1)
 for period in [15,30,60]:
  f=old/product/(contract+f'_{period}m.csv')
  if not f.exists():continue
  o=pd.read_csv(f,parse_dates=['datetime']).set_index('datetime')
  n=x.resample(f'{period}min',origin='start_day',label='left',closed='left').agg(dict(open='first',high='max',low='min',close='last',volume='sum',open_oi='first',close_oi='last')).dropna()
  # Exclude possible partial bins on each download edge, not internal breaks.
  n=n.loc[(n.index>=x.index.min())&(n.index+pd.Timedelta(minutes=period)<=x.index.max()+pd.Timedelta(minutes=1))]
  index=n.index.intersection(o.index);a=n.loc[index,fields].to_numpy();b=o.loc[index,fields].to_numpy();d=np.abs(a-b)
  neq=~np.isclose(a,b,atol=1e-8,rtol=1e-10)
  rows.append(dict(product=product,contract=contract,period=period,old_path=str(f),new_path=r['path'],common_rows=len(index),rows_different=int(neq.any(1).sum()),different_by_field=dict(zip(fields,neq.sum(0).tolist())),max_abs_by_field=dict(zip(fields,d.max(0).tolist()))))
report={'scope':'deterministic first filename per shared product whose new interval contains 2026-06-01; nominal wall-clock aggregation; no session completeness qualification; no feature rebuilding', 'cases':rows,'common_rows':sum(r['common_rows'] for r in rows),'rows_different':sum(r['rows_different'] for r in rows),'products':sorted({r['product'] for r in rows})}
Path('reports/tqdata_audit_20261124_final/old_overlap_spotcheck.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='cases'})
print([r for r in rows if r['rows_different']])
