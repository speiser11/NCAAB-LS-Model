"""
Test impact of the KenPom luck regression adjustment on model accuracy.

Fetches today's kenpom-teams.json from the Gist, applies the same luck
adjustment logic used in the JS model to historical game predictions, then
compares:
  - Baseline MAE (no luck adjustment)
  - Adjusted MAE (with luck regression)
  - ATS win rate at different edge thresholds, with and without luck adj
  - Breakdown by adjustment magnitude

⚠ IMPORTANT CAVEAT:
  We only have today's luck values — not historical snapshots per game.
  Luck accumulates and shifts throughout the season, so applying current
  luck to old games is inherently lookahead-biased.  Results are directional
  only.  A neutral result here does NOT mean the adjustment is harmful in
  live use (where luck is current); a harmful result IS a warning sign.

Requires: dataset.csv (run optimize.py first)
"""
import re
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data')
GIST_URL = ('https://gist.githubusercontent.com/speiser11/'
            '44dd1f1464e3d7e2689b25ba758d4ea9/raw/kenpom-teams.json')


def norm_team(name: str) -> str:
    """Mirror of JS normTeam() — strip punctuation, handle St./State."""
    s = name.lower()
    s = re.sub(r'\s*\([^)]{0,6}\)\s*', ' ', s)
    s = re.sub(r"[^a-z0-9\s'&.-]", '', s)
    s = re.sub(r'([a-z])\.([a-z])\.', r'\1\2', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r' st\.$', ' state', s)
    s = re.sub(r'\bst\b', 'state', s)
    return s


def match_team(full_name: str, key_set: set) -> str | None:
    """Mirror of JS matchTeam(): normalize then strip trailing words (mascot)."""
    norm = norm_team(full_name)
    if norm in key_set:
        return norm
    words = norm.split()
    for length in range(len(words) - 1, 0, -1):
        shorter = ' '.join(words[:length])
        if shorter in key_set:
            return shorter
    return None

AVG_D       = 106.09
SCORE_SCALE = 1.04
HOME_COURT  = 3.5
LUCK_SCALE  = 25     # pts per luck unit — mirrors JS model
LUCK_CAP    = 3      # max adjustment in either direction
LUCK_MIN    = 0.1    # ignore adjustments smaller than this
BREAKEVEN   = 0.5238

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'dataset.csv')
print(f'Loaded {len(df):,} games')

print('Fetching kenpom-teams.json from Gist...')
r = requests.get(GIST_URL, timeout=15)
r.raise_for_status()
kenpom = r.json()
luck_map = {k: v['luck'] for k, v in kenpom.items() if 'luck' in v}
print(f'  {len(kenpom)} teams in file, {len(luck_map)} have luck values\n')

# ── Baseline predictions ──────────────────────────────────
poss    = np.sqrt(df['h_adjT'] * df['a_adjT'])
h_score = df['h_adjO'] * (df['a_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
a_score = df['a_adjO'] * (df['h_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
hc      = np.where(df['neutral'].astype(bool), 0.0, HOME_COURT)
pred_base = (h_score - a_score) + hc

# ── Match dataset names → KenPom keys ────────────────────
# dataset.csv uses full ESPN names ("North Carolina Tar Heels"); KenPom keys
# are clean names ("north carolina"). Use the same word-stripping matchTeam logic.
kp_keys = set(luck_map.keys())
h_key = df['home'].map(lambda n: match_team(n, kp_keys))
a_key = df['away'].map(lambda n: match_team(n, kp_keys))

# ── Apply luck adjustment ─────────────────────────────────
h_luck = h_key.map(lambda k: luck_map[k] if isinstance(k, str) else 0.0)
a_luck = a_key.map(lambda k: luck_map[k] if isinstance(k, str) else 0.0)

raw_adj  = (a_luck - h_luck) * LUCK_SCALE
luck_adj = raw_adj.clip(-LUCK_CAP, LUCK_CAP)
luck_adj = luck_adj.where(luck_adj.abs() >= LUCK_MIN, 0.0)  # suppress tiny adjustments

pred_adj = pred_base + luck_adj

# Coverage stats
h_matched = h_key.notna()
a_matched = a_key.notna()
both_matched = h_matched & a_matched
any_matched  = h_matched | a_matched
adj_applied  = luck_adj.abs() >= LUCK_MIN

print(f'  Team match rate:')
print(f'    Both teams in KenPom:       {both_matched.sum():,} / {len(df):,} games ({both_matched.mean()*100:.1f}%)')
print(f'    At least one in KenPom:     {any_matched.sum():,} / {len(df):,} games ({any_matched.mean()*100:.1f}%)')
print(f'    Adj applied (|adj|≥{LUCK_MIN}):    {adj_applied.sum():,} games ({adj_applied.mean()*100:.1f}%)')
if adj_applied.sum():
    print(f'    Avg adjustment (applied):   {luck_adj[adj_applied].abs().mean():.2f} pts')
print()

# ── Overall accuracy ──────────────────────────────────────
err_base = df['actual_margin'] - pred_base
err_adj  = df['actual_margin'] - pred_adj

print('=' * 64)
print('PREDICTION ACCURACY — baseline vs luck-adjusted')
print('=' * 64)
print(f'  {"":32}  {"Baseline":>9}  {"Adjusted":>9}  {"Delta":>7}')
print(f'  {"MAE (all games)":32}  {err_base.abs().mean():>9.3f}  {err_adj.abs().mean():>9.3f}  {err_adj.abs().mean()-err_base.abs().mean():>+7.3f}')
print(f'  {"RMSE (all games)":32}  {np.sqrt((err_base**2).mean()):>9.3f}  {np.sqrt((err_adj**2).mean()):>9.3f}  {np.sqrt((err_adj**2).mean())-np.sqrt((err_base**2).mean()):>+7.3f}')
print()

# ── Subset: games where adjustment was applied ────────────
if adj_applied.sum() >= 10:
    bm = err_base[adj_applied].abs().mean()
    am = err_adj[adj_applied].abs().mean()
    print(f'  {"MAE — adj applied subset":32}  {bm:>9.3f}  {am:>9.3f}  {am-bm:>+7.3f}')
    print()

# ── Breakdown by adjustment magnitude ─────────────────────
print('Breakdown by luck adjustment magnitude:')
print(f'  {"Range":16}  {"n":>6}  {"Base MAE":>9}  {"Adj MAE":>9}  {"Delta":>7}')
for lo, hi, label in [(0.1, 1.0, '0.1–1.0 pts'), (1.0, 2.0, '1.0–2.0 pts'), (2.0, 4.0, '2.0+ pts')]:
    mask = (luck_adj.abs() >= lo) & (luck_adj.abs() < hi)
    if mask.sum() < 10:
        continue
    bm = err_base[mask].abs().mean()
    am = err_adj[mask].abs().mean()
    print(f'  {label:16}  {mask.sum():>6}  {bm:>9.3f}  {am:>9.3f}  {am-bm:>+7.3f}')
print()

# ── Lucky vs unlucky team direction check ─────────────────
# Do teams with positive luck (overperformers) actually tend to be overestimated?
print('Direction check — lucky teams (home perspective):')
print('  If luck signal is real: highly-lucky home teams should show positive')
print('  model error (model overestimates them = actual < predicted)')
print()
print(f'  {"h_luck bucket":20}  {"n":>6}  {"avg actual err":>14}  {"avg pred err base":>18}')
for lo, hi, label in [(0.05, 1.0, 'Home lucky (>0.05)'), (-1.0, -0.05, 'Home unlucky (<-0.05)'), (-0.02, 0.02, 'Home neutral')]:
    mask = (h_luck >= lo) & (h_luck < hi)
    if mask.sum() < 10:
        continue
    avg_err = err_base[mask].mean()  # positive = model underestimated home team
    print(f'  {label:20}  {mask.sum():>6}  {avg_err:>+14.3f}  (positive = model underestimated home)')
print()

# ── ATS win rate ──────────────────────────────────────────
print('=' * 64)
print('ATS WIN RATE vs VEGAS SPREAD — base vs luck-adjusted')
print('=' * 64)

sp = df[df['vegas_spread'].notna()].copy()
sp['edge_base'] = pred_base[sp.index] + sp['vegas_spread']
sp['edge_adj']  = pred_adj[sp.index]  + sp['vegas_spread']
sp['cover']     = sp['actual_margin'] + sp['vegas_spread']
sp = sp[sp['cover'] != 0]

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
print('=' * 64)
print('SUMMARY')
print('=' * 64)
mae_delta = err_adj.abs().mean() - err_base.abs().mean()
print(f'  Overall MAE change:       {mae_delta:>+.3f} pts  (−=better)')
if adj_applied.sum():
    sub_delta = err_adj[adj_applied].abs().mean() - err_base[adj_applied].abs().mean()
    print(f'  MAE change (adj subset):  {sub_delta:>+.3f} pts')
print(f'  Games affected:           {adj_applied.sum():,} ({adj_applied.mean()*100:.1f}%)')
print()
print('⚠ Caveat: uses today\'s luck values on historical games (lookahead bias).')
print('  Luck snapshots at game time would be needed for an unbiased test.')
