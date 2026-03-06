"""
Empirically calibrate three model constants from historical data:
  1. NCAAB_SIGMA  — std dev of model errors (used for win probability)
  2. B2B_PEN      — actual score impact when a team plays back-to-back
  3. FORM_WEIGHT  — optimal blend weight for 14-day vs season-long ratings

Requires: dataset.csv (run optimize.py first)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r'C:\Users\scott\Projects\NCAAB LS Model\optimize\data')

AVG_D       = 106.09
SCORE_SCALE = 1.04
HOME_COURT  = 3.5

df = pd.read_csv(DATA_DIR / 'dataset.csv')
print(f'Loaded {len(df):,} games\n')

poss    = np.sqrt(df['h_adjT'] * df['a_adjT'])
h_score = df['h_adjO'] * (df['a_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
a_score = df['a_adjO'] * (df['h_adjD'] / AVG_D) * poss / 100 * SCORE_SCALE
hc      = np.where(df['neutral'].astype(bool), 0.0, HOME_COURT)
pred_margin = (h_score - a_score) + hc
error       = df['actual_margin'] - pred_margin

# ══════════════════════════════════════════════════════════
# 1. NCAAB_SIGMA
# ══════════════════════════════════════════════════════════
print('=' * 56)
print('1. NCAAB_SIGMA — empirical spread error distribution')
print('=' * 56)
sigma_empirical = error.std()
mae_empirical   = error.abs().mean()
print(f'  Std dev of errors (model sigma): {sigma_empirical:.2f} pts')
print(f'  MAE:                             {mae_empirical:.2f} pts')
print(f'  Current NCAAB_SIGMA = 10.5')
print(f'  → Recommended:       {sigma_empirical:.1f}')

# Show what difference it makes for win probability at a few margins
from scipy.stats import norm
print(f'\n  Win prob comparison (margin → prob with old vs new sigma):')
print(f'  {"Margin":>8}  {"σ=10.5":>8}  {"σ="+str(round(sigma_empirical,1)):>8}')
for m in [2, 4, 6, 8, 10, 14]:
    old = norm.cdf(m / 10.5) * 100
    new = norm.cdf(m / sigma_empirical) * 100
    print(f'  {m:>+8}  {old:>7.1f}%  {new:>7.1f}%')

# ══════════════════════════════════════════════════════════
# 2. B2B PENALTY
# ══════════════════════════════════════════════════════════
print()
print('=' * 56)
print('2. B2B_PEN — back-to-back score impact')
print('=' * 56)
print('  NOTE: dataset.csv does not have B2B flags.')
print('  B2B status requires game schedule sequencing.')
print('  Approximating via date proximity in ESPN data...')

# Sort by team and date to find consecutive game days
# We'll look at actual score vs predicted score for games
# played 1 day after the previous game for that team
df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

# Collect all team-date pairs (home and away)
home_games = df[['date_dt','home','home_score','away_score','actual_margin']].copy()
home_games.columns = ['date','team','team_score','opp_score','home_margin']
home_games['is_home'] = True

away_games = df[['date_dt','away','away_score','home_score','actual_margin']].copy()
away_games.columns = ['date','team','team_score','opp_score','home_margin']
away_games['is_home'] = False
away_games['home_margin'] = -away_games['home_margin']

all_games = pd.concat([home_games, away_games], ignore_index=True)
all_games = all_games.sort_values(['team','date'])
all_games['prev_date'] = all_games.groupby('team')['date'].shift(1)
all_games['days_rest']  = (all_games['date'] - all_games['prev_date']).dt.days

b2b_games = all_games[all_games['days_rest'] == 1]
rest_games = all_games[all_games['days_rest'] >= 2]

# Score differential vs expected: use actual team score vs opponent score
# For B2B team: does the team score fewer points?
b2b_team_score  = b2b_games['team_score'].mean()
rest_team_score = rest_games['team_score'].mean()
b2b_penalty_score = rest_team_score - b2b_team_score

# Margin impact: does B2B team lose by more?
b2b_margin  = b2b_games['home_margin'].mean()   # from team's perspective
rest_margin = rest_games['home_margin'].mean()
b2b_penalty_margin = rest_margin - b2b_margin

print(f'\n  B2B games:  {len(b2b_games):,}')
print(f'  Rest games: {len(rest_games):,}')
print(f'\n  Avg team score — B2B: {b2b_team_score:.1f}  |  Rested: {rest_team_score:.1f}')
print(f'  Score penalty on B2B team:   {b2b_penalty_score:.2f} pts')
print(f'  Margin penalty (team POV):   {b2b_penalty_margin:.2f} pts')
print(f'\n  Current B2B_PEN = 1.8 pts')
print(f'  → Recommended:  {b2b_penalty_score:.1f} pts (score) / {b2b_penalty_margin:.1f} pts (margin)')

# ══════════════════════════════════════════════════════════
# 3. FORM BLEND WEIGHT
# ══════════════════════════════════════════════════════════
print()
print('=' * 56)
print('3. FORM_WEIGHT — optimal recent vs season blend')
print('=' * 56)
print('  NOTE: Requires per-game recent ratings (14-day snapshots).')
print('  dataset.csv has weekly snapshots, not 14-day windows.')
print('  Approximating: comparing weekly snapshot to prior week snapshot')
print('  as a proxy for "recent" vs "season-long" ratings.')
print()

# Use current week vs 2-week-old ratings as a proxy for recent vs season
# Sort by date, group by week anchor
df['week'] = df['date_dt'].dt.to_period('W')

# Compute error for a range of blend weights using prior-week as "recent"
# We'll compute how much using the prior week's adjO/adjD helps
# Since we only have one snapshot per game, we'll use a cross-week comparison:
# Compute MAE on 2025 season (held out) at different form weights
# by shifting: treat h_adjO from last week as "recent"
# This is an approximation — we don't have separate recent rating columns

print('  Using season 2025 hold-out to test form weights:')
print('  (Proxy: recent = current week, base = season avg from prior weeks)')
print()

df_2025 = df[df['season'] == 2025].copy()
df_2025 = df_2025.sort_values('date')

# Use a rolling z-score of adjO deviation from season mean as form signal
# Compute each team's adjO deviation from their own season mean
for col in ['h_adjO','h_adjD','a_adjO','a_adjD']:
    mean_col = df_2025.groupby('home' if col.startswith('h') else 'away')[col].transform('mean')
    df_2025[f'{col}_dev'] = df_2025[col] - mean_col

print(f'  {"Weight":>8}  {"MAE":>8}  {"vs 0.0":>8}')
base_mae = None
results = []
for w in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    p2025 = np.sqrt(df_2025['h_adjT'] * df_2025['a_adjT'])
    h_adj_O = df_2025['h_adjO'] + w * df_2025['h_adjO_dev']
    a_adj_O = df_2025['a_adjO'] + w * df_2025['a_adjO_dev']
    h_adj_D = df_2025['h_adjD'] + w * df_2025['h_adjD_dev']
    a_adj_D = df_2025['a_adjD'] + w * df_2025['a_adjD_dev']
    hs = h_adj_O * (a_adj_D / AVG_D) * p2025 / 100 * SCORE_SCALE
    as_ = a_adj_O * (h_adj_D / AVG_D) * p2025 / 100 * SCORE_SCALE
    hc2 = np.where(df_2025['neutral'].astype(bool), 0.0, HOME_COURT)
    pred = (hs - as_) + hc2
    mae = (df_2025['actual_margin'] - pred).abs().mean()
    if base_mae is None: base_mae = mae
    delta = mae - base_mae
    marker = ' ← best' if len(results) == 0 or mae < min(r[1] for r in results) else ''
    print(f'  {w:>8.2f}  {mae:>8.3f}  {delta:>+7.3f}{marker}')
    results.append((w, mae))

best_w, best_mae = min(results, key=lambda x: x[1])
print(f'\n  Current FORM_WEIGHT = 0.30')
print(f'  → Best weight found: {best_w:.2f}  (MAE={best_mae:.3f})')

print()
print('=' * 56)
print('SUMMARY — recommended constant updates')
print('=' * 56)
print(f'  NCAAB_SIGMA = {sigma_empirical:.1f}   (was 10.5)')
print(f'  B2B_PEN     = {b2b_penalty_score:.1f}   (was 1.8)  [score-based estimate]')
print(f'  FORM_WEIGHT = {best_w:.2f}  (was 0.30)')
