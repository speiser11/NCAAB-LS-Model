"""
Compute model win rate by edge size for spreads and totals.
This tells you what thresholds actually mean in terms of predictive confidence.
Breakeven ATS at -110 odds = 52.38%
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data\dataset.csv')

AVG_D      = 106.09
SCORE_SCALE = 1.04
TOT_SCALE   = 0.847
TOT_OFFSET  = 18.7

poss    = np.sqrt(df['h_adjT'] * df['a_adjT'])
h_score = df['h_adjO'] * (df['a_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
a_score = df['a_adjO'] * (df['h_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
hc      = np.where(df['neutral'].astype(bool), 0.0, 3.5)

pred_margin = (h_score - a_score) + hc
pred_total  = (h_score + a_score) * TOT_SCALE + TOT_OFFSET

BREAKEVEN = 0.5238

print('=' * 62)
print('SPREAD — ATS win rate by model edge size')
print(f'{"Edge":>6}  {"Games":>6}  {"ATS%":>6}  {"vs Breakeven":>13}  Signal')
print('-' * 62)

sp = df[df['vegas_spread'].notna()].copy()
sp['edge'] = pred_margin[sp.index] + sp['vegas_spread']   # +ve = model backs home
sp['cover'] = sp['actual_margin'] + sp['vegas_spread']    # +ve = home covered
sp = sp[sp['cover'] != 0]                                 # exclude pushes
sp['backed_home'] = sp['edge'] > 0
sp['correct'] = sp['backed_home'] == (sp['cover'] > 0)

buckets = [1, 2, 3, 4, 5, 6, 7, 8, 10]
for lo, hi in zip(buckets, buckets[1:] + [99]):
    mask = sp['edge'].abs().between(lo, hi, inclusive='left')
    sub  = sp[mask]
    if len(sub) < 30:
        continue
    rate = sub['correct'].mean()
    sig  = '★★★' if rate >= 0.56 else '★★' if rate >= 0.54 else '★' if rate >= BREAKEVEN else '—'
    print(f'{lo:>3}-{hi:<3}  {len(sub):>6}  {rate*100:>5.1f}%  {(rate-BREAKEVEN)*100:>+10.1f}pp  {sig}')

mask_all = sp['edge'].abs() >= 1
rate_all = sp[mask_all]['correct'].mean()
print(f'{"All":>6}  {mask_all.sum():>6}  {rate_all*100:>5.1f}%  {(rate_all-BREAKEVEN)*100:>+10.1f}pp')

print()
print('=' * 62)
print('TOTAL — O/U win rate by model edge size')
print(f'{"Edge":>6}  {"Games":>6}  {"OU%":>6}  {"vs Breakeven":>13}  Signal')
print('-' * 62)

tot = df[df['vegas_total'].notna()].copy()
tot['pred_tot'] = pred_total[tot.index]
tot['edge']     = tot['pred_tot'] - tot['vegas_total']    # +ve = model leans over
tot['actual_tot'] = tot['actual_total']
tot = tot[tot['actual_tot'] != tot['vegas_total']]        # exclude pushes
tot['backed_over'] = tot['edge'] > 0
tot['correct'] = tot['backed_over'] == (tot['actual_tot'] > tot['vegas_total'])

for lo, hi in zip(buckets, buckets[1:] + [99]):
    mask = tot['edge'].abs().between(lo, hi, inclusive='left')
    sub  = tot[mask]
    if len(sub) < 30:
        continue
    rate = sub['correct'].mean()
    sig  = '★★★' if rate >= 0.56 else '★★' if rate >= 0.54 else '★' if rate >= BREAKEVEN else '—'
    print(f'{lo:>3}-{hi:<3}  {len(sub):>6}  {rate*100:>5.1f}%  {(rate-BREAKEVEN)*100:>+10.1f}pp  {sig}')

mask_all = tot['edge'].abs() >= 1
rate_all = tot[mask_all]['correct'].mean()
print(f'{"All":>6}  {mask_all.sum():>6}  {rate_all*100:>5.1f}%  {(rate_all-BREAKEVEN)*100:>+10.1f}pp')

print()
print(f'Breakeven at -110 odds: {BREAKEVEN*100:.2f}%')
print('★ = above breakeven  ★★ = +1.5pp  ★★★ = +3.5pp edge')
