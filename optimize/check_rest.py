"""
Test impact of rest-differential adjustments on model accuracy.

Reconstructs days-since-last-game per team from the historical schedule,
applies the same rest adjustment logic used in the JS model, then compares:
  - Baseline MAE (no rest adjustment)
  - Adjusted MAE (with rest differential)
  - ATS win rate at different edge thresholds, with and without rest adj

Requires: dataset.csv (run optimize.py first)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data')

AVG_D       = 106.09
SCORE_SCALE = 1.04
HOME_COURT  = 3.5
NCAAB_SIGMA = 12.5
B2B_PEN     = 1.8
REST_PTS_PER_DAY = 0.15   # pts per day of rest advantage (non-B2B) — tested, adds noise (+0.033 MAE), removed from model
REST_DIFF_MIN    = 2       # minimum day gap to apply differential
REST_DIFF_CAP    = 1.5     # max pts from rest differential
BREAKEVEN = 0.5238

df = pd.read_csv(DATA_DIR / 'dataset.csv')
print(f'Loaded {len(df):,} games\n')

# ── Baseline predictions ──────────────────────────────────
poss    = np.sqrt(df['h_adjT'] * df['a_adjT'])
h_score = df['h_adjO'] * (df['a_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
a_score = df['a_adjO'] * (df['h_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
hc      = np.where(df['neutral'].astype(bool), 0.0, HOME_COURT)
pred_margin_base = (h_score - a_score) + hc

# ── Compute days since last game for each team ────────────
df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

# Build per-team game log (home + away)
home_games = df[['date_dt', 'home']].copy().rename(columns={'home': 'team'})
away_games = df[['date_dt', 'away']].copy().rename(columns={'away': 'team'})
all_games  = pd.concat([home_games, away_games], ignore_index=True)
all_games  = all_games.sort_values(['team', 'date_dt']).reset_index(drop=True)
all_games['prev_date'] = all_games.groupby('team')['date_dt'].shift(1)
all_games['days_rest'] = (all_games['date_dt'] - all_games['prev_date']).dt.days.fillna(7).clip(upper=7)

# Merge back to games as h_days_rest / a_days_rest
h_rest = all_games.rename(columns={'team': 'home', 'days_rest': 'h_days_rest'}).drop(columns=['prev_date'])
a_rest = all_games.rename(columns={'team': 'away', 'days_rest': 'a_days_rest'}).drop(columns=['prev_date'])

df = df.merge(h_rest[['date_dt', 'home', 'h_days_rest']], on=['date_dt', 'home'], how='left')
df = df.merge(a_rest[['date_dt', 'away', 'a_days_rest']], on=['date_dt', 'away'], how='left')
df['h_days_rest'] = df['h_days_rest'].fillna(7).clip(upper=7)
df['a_days_rest'] = df['a_days_rest'].fillna(7).clip(upper=7)

h_b2b = df['h_days_rest'] == 1
a_b2b = df['a_days_rest'] == 1

# ── Apply rest adjustment ─────────────────────────────────
rest_adj = np.zeros(len(df))

# B2B penalty
rest_adj -= h_b2b.astype(float) * B2B_PEN
rest_adj += a_b2b.astype(float) * B2B_PEN

# Rest differential (non-B2B only)
both_rested = ~h_b2b & ~a_b2b
diff = df['h_days_rest'] - df['a_days_rest']
big_gap = both_rested & (diff.abs() >= REST_DIFF_MIN)
rest_adj += np.where(
    big_gap,
    np.clip(diff * REST_PTS_PER_DAY, -REST_DIFF_CAP, REST_DIFF_CAP),
    0.0
)

pred_margin_adj = pred_margin_base + rest_adj

# ── Summary stats ─────────────────────────────────────────
error_base = df['actual_margin'] - pred_margin_base
error_adj  = df['actual_margin'] - pred_margin_adj

print('=' * 62)
print('PREDICTION ACCURACY — baseline vs rest-adjusted')
print('=' * 62)
print(f'  {"":30}  {"Baseline":>9}  {"Adjusted":>9}  {"Delta":>7}')
print(f'  {"MAE (all games)":30}  {error_base.abs().mean():>9.3f}  {error_adj.abs().mean():>9.3f}  {error_adj.abs().mean()-error_base.abs().mean():>+7.3f}')
print(f'  {"RMSE (all games)":30}  {np.sqrt((error_base**2).mean()):>9.3f}  {np.sqrt((error_adj**2).mean()):>9.3f}  {np.sqrt((error_adj**2).mean())-np.sqrt((error_base**2).mean()):>+7.3f}')
print()

# ── B2B subset performance ───────────────────────────────
b2b_any = h_b2b | a_b2b
print(f'  {"B2B games: n":30}  {b2b_any.sum():,}')
print(f'  {"MAE — B2B games (base)":30}  {error_base[b2b_any].abs().mean():>9.3f}')
print(f'  {"MAE — B2B games (adj)":30}  {error_adj[b2b_any].abs().mean():>9.3f}  Δ={error_adj[b2b_any].abs().mean()-error_base[b2b_any].abs().mean():>+.3f}')
print()

# ── B2B breakdown ─────────────────────────────────────────
print('  B2B breakdown: actual vs predicted margin direction')
print(f'  {"":30}  {"n":>6}  {"actual_margin":>14}  {"pred_base":>10}  {"pred_adj":>10}')
for label, mask in [('Home on B2B', h_b2b & ~a_b2b), ('Away on B2B', a_b2b & ~h_b2b), ('Both on B2B', h_b2b & a_b2b)]:
    if mask.sum() < 5: continue
    print(f'  {label:30}  {mask.sum():>6}  {df.loc[mask,"actual_margin"].mean():>+14.2f}  {pred_margin_base[mask].mean():>+10.2f}  {pred_margin_adj[mask].mean():>+10.2f}')
print()

# ── Rest differential subset ──────────────────────────────
print(f'  Rest-diff games (≥{REST_DIFF_MIN}d gap, non-B2B): n={big_gap.sum():,}')
if big_gap.sum() > 0:
    print(f'  {"MAE — rest-diff games (base)":30}  {error_base[big_gap].abs().mean():>9.3f}')
    print(f'  {"MAE — rest-diff games (adj)":30}  {error_adj[big_gap].abs().mean():>9.3f}  Δ={error_adj[big_gap].abs().mean()-error_base[big_gap].abs().mean():>+.3f}')
print()

# ── ATS win rate (games with Vegas spread) ───────────────
print('=' * 62)
print('ATS WIN RATE vs VEGAS SPREAD — base vs adjusted model')
print('=' * 62)

sp = df[df['vegas_spread'].notna()].copy()
sp['edge_base'] = pred_margin_base[sp.index] + sp['vegas_spread']
sp['edge_adj']  = pred_margin_adj[sp.index]  + sp['vegas_spread']
sp['cover']     = sp['actual_margin'] + sp['vegas_spread']
sp = sp[sp['cover'] != 0]  # drop pushes

def ats_rate(edge_col, threshold):
    mask = sp[edge_col].abs() >= threshold
    sub  = sp[mask]
    if len(sub) < 30: return None, 0
    correct = (sp.loc[mask, edge_col] > 0) == (sp.loc[mask, 'cover'] > 0)
    return correct.mean(), len(sub)

print(f'  {"Edge ≥":>7}  {"Base n":>7}  {"Base%":>7}  {"Adj n":>7}  {"Adj%":>7}  {"Δpp":>6}')
for thr in [3, 4, 5, 6, 7, 8, 10]:
    r_base, n_base = ats_rate('edge_base', thr)
    r_adj,  n_adj  = ats_rate('edge_adj',  thr)
    if r_base is None: continue
    delta = (r_adj - r_base) * 100
    sig = '★' if r_adj >= BREAKEVEN else ''
    print(f'  {thr:>7}  {n_base:>7}  {r_base*100:>6.1f}%  {n_adj:>7}  {r_adj*100:>6.1f}%  {delta:>+5.1f}  {sig}')

print()
print('=' * 62)
print('SUMMARY')
print('=' * 62)
mae_delta = error_adj.abs().mean() - error_base.abs().mean()
print(f'  Overall MAE change:     {mae_delta:>+.3f} pts  (−=better)')
print(f'  B2B game MAE change:    {error_adj[b2b_any].abs().mean()-error_base[b2b_any].abs().mean():>+.3f} pts')
if big_gap.sum() > 0:
    print(f'  Rest-diff MAE change:   {error_adj[big_gap].abs().mean()-error_base[big_gap].abs().mean():>+.3f} pts')
print(f'  B2B games in dataset:   {b2b_any.sum():,} ({b2b_any.mean()*100:.1f}% of all games)')
print(f'  Rest-diff games:        {big_gap.sum():,} ({big_gap.mean()*100:.1f}% of all games)')
print()
print('Note: injury impact cannot be tested here (no historical injury data in dataset.csv)')
