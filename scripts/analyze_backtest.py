"""
analyze_backtest.py

Regression analysis on mlb_backtest.csv to evaluate which model factors
are actually predictive and where the model needs calibration.

Usage:
  cd "NCAAB LS Model"
  py scripts/analyze_backtest.py

Output: printed report (no files written)
"""

import csv, math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('mlb_backtest.csv')
print(f'Loaded {len(df)} rows\n')

def pct(n, d):
    return f'{n}/{d} = {n/d*100:.1f}%' if d else 'n/a'

# ── 1. Calibration: model win prob vs actual win rate ─────────────────────────
print('=' * 60)
print('1. ML WIN PROB CALIBRATION (bucketed)')
print('=' * 60)
ml_df = df[df['ml_correct'].isin([True, False])].copy()
ml_df['model_home_win_pct'] = pd.to_numeric(ml_df['model_home_win_pct'], errors='coerce')
ml_df['ml_correct_int'] = ml_df['ml_correct'].map({True: 1, False: 0})

# Home-favored vs away-favored
home_fav = ml_df[ml_df['model_home_win_pct'] >= 50]
away_fav = ml_df[ml_df['model_home_win_pct'] < 50]
hf_correct = home_fav[home_fav['ml_correct'] == True]
af_correct = away_fav[away_fav['ml_correct'] == True]
print(f'Home-favored games:  {pct(len(hf_correct), len(home_fav))} correct')
print(f'Away-favored games:  {pct(len(af_correct), len(away_fav))} correct')

# Bucket by model confidence
buckets = [(30,45,'30-45%'),(45,55,'45-55%'),(55,65,'55-65%'),(65,75,'65-75%')]
print('\nBucket      | games | model_avg | actual_win%')
for lo, hi, label in buckets:
    sub = ml_df[(ml_df['model_home_win_pct'] >= lo) & (ml_df['model_home_win_pct'] < hi)]
    if len(sub) == 0:
        continue
    actual_rate = sub['ml_correct_int'].mean() * 100
    model_avg   = sub['model_home_win_pct'].mean()
    print(f'{label:12s}| {len(sub):5d} | {model_avg:8.1f}% | {actual_rate:.1f}%')

# ── 2. Logistic regression: what predicts ML outcome ─────────────────────────
print('\n' + '=' * 60)
print('2. LOGISTIC REGRESSION: factors predicting ML correct/wrong')
print('=' * 60)

feat_cols = [
    'home_pitcher_quality', 'away_pitcher_quality',
    'home_bullpen_score',   'away_bullpen_score',
    'home_offense_score',   'away_offense_score',
    'park_factor',          'model_run_diff',
    'ml_imp_diff_pct',
]

lr_df = ml_df.copy()
for c in feat_cols:
    lr_df[c] = pd.to_numeric(lr_df[c], errors='coerce')
lr_df = lr_df.dropna(subset=feat_cols + ['ml_correct_int'])

X = lr_df[feat_cols].values
y = lr_df['ml_correct_int'].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(Xs, y)

print(f'Training accuracy: {clf.score(Xs, y)*100:.1f}%')
print(f'Log-loss:          {log_loss(y, clf.predict_proba(Xs)):.4f}')
print(f'Baseline log-loss: {log_loss(y, [[y.mean(), 1-y.mean()]]*len(y)):.4f}')
print(f'\nCoefficients (standardized — larger abs = stronger predictor):')
coefs = sorted(zip(feat_cols, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for name, coef in coefs:
    direction = '+' if coef > 0 else '-'
    print(f'  {direction}  {abs(coef):.4f}  {name}')

# ── 3. Total prediction error analysis ───────────────────────────────────────
print('\n' + '=' * 60)
print('3. TOTAL PREDICTION ERROR')
print('=' * 60)

tot_df = df.copy()
for c in ['model_total', 'actual_total', 'vegas_total']:
    tot_df[c] = pd.to_numeric(tot_df[c], errors='coerce')
tot_df = tot_df.dropna(subset=['model_total', 'actual_total'])

tot_df['model_error']  = tot_df['model_total'] - tot_df['actual_total']
tot_df['model_abs_err'] = tot_df['model_error'].abs()

print(f'Games with model total: {len(tot_df)}')
print(f'Mean error (+ = model high): {tot_df["model_error"].mean():.2f} runs')
print(f'Mean absolute error:         {tot_df["model_abs_err"].mean():.2f} runs')
print(f'Model over-predicts by 1+:   {(tot_df["model_error"] >= 1).sum()} games')
print(f'Model under-predicts by 1+:  {(tot_df["model_error"] <= -1).sum()} games')

# vs Vegas
has_vegas = tot_df.dropna(subset=['vegas_total'])
has_vegas = has_vegas[has_vegas['vegas_total'] > 0]
has_vegas['vegas_error'] = has_vegas['vegas_total'] - has_vegas['actual_total']
has_vegas['vegas_abs_err'] = has_vegas['vegas_error'].abs()
print(f'\nVs Vegas (games with lines: {len(has_vegas)}):')
print(f'  Model MAE:  {has_vegas["model_abs_err"].mean():.2f}')
print(f'  Vegas MAE:  {has_vegas["vegas_abs_err"].mean():.2f}')
print(f'  Model bias: {has_vegas["model_error"].mean():.2f} (+ = model too high)')
print(f'  Vegas bias: {has_vegas["vegas_error"].mean():.2f} (+ = Vegas too high)')

# Direction accuracy by total_diff size
print('\nTotal edge accuracy by model_diff vs Vegas:')
edge_df = has_vegas.dropna(subset=['model_total','vegas_total'])
edge_df['total_diff'] = edge_df['model_total'] - edge_df['vegas_total']
edge_df['model_dir'] = edge_df['total_diff'].apply(lambda x: 'over' if x > 0 else 'under')
edge_df['ou_result'] = edge_df['actual_total'].combine(edge_df['vegas_total'],
    lambda a, v: 'over' if a > v else ('under' if a < v else 'push'))
edge_df = edge_df[edge_df['ou_result'] != 'push']
edge_df['dir_correct'] = edge_df['model_dir'] == edge_df['ou_result']

for cutoff in [1.0, 1.5, 2.0, 2.5, 3.0]:
    sub = edge_df[edge_df['total_diff'].abs() >= cutoff]
    if len(sub) == 0:
        continue
    hits = sub['dir_correct'].sum()
    over_sub  = sub[sub['model_dir'] == 'over']
    under_sub = sub[sub['model_dir'] == 'under']
    o_hits = over_sub['dir_correct'].sum()
    u_hits = under_sub['dir_correct'].sum()
    print(f'  |diff| >= {cutoff}: {pct(hits, len(sub))}  '
          f'(over: {pct(o_hits, len(over_sub))}  under: {pct(u_hits, len(under_sub))})')

# ── 4. Linear regression: what drives total error ─────────────────────────────
print('\n' + '=' * 60)
print('4. LINEAR REGRESSION: factors driving total over/under-prediction')
print('=' * 60)

tot_feat_cols = [
    'home_pitcher_quality', 'away_pitcher_quality',
    'home_bullpen_score',   'away_bullpen_score',
    'home_offense_score',   'away_offense_score',
    'park_factor',
]
reg_df = tot_df.copy()
for c in tot_feat_cols:
    reg_df[c] = pd.to_numeric(reg_df[c], errors='coerce')
reg_df = reg_df.dropna(subset=tot_feat_cols + ['model_error'])

Xt = reg_df[tot_feat_cols].values
yt = reg_df['model_error'].values
scaler_t = StandardScaler()
Xts = scaler_t.fit_transform(Xt)
lm = LinearRegression()
lm.fit(Xts, yt)
r2 = lm.score(Xts, yt)
print(f'R^2: {r2:.4f}  (how much of model error is explained by inputs)')
print(f'\nCoefficients (+ = that factor makes model predict TOO HIGH):')
coefs_t = sorted(zip(tot_feat_cols, lm.coef_), key=lambda x: abs(x[1]), reverse=True)
for name, coef in coefs_t:
    direction = '+' if coef > 0 else '-'
    print(f'  {direction}  {abs(coef):.4f}  {name}')

# ── 5. Over/Under bias check ──────────────────────────────────────────────────
print('\n' + '=' * 60)
print('5. OVER/UNDER BIAS (model direction hit rates)')
print('=' * 60)

ou_df = has_vegas[has_vegas['total_ou_result'].isin(['over','under'])].copy()
ou_df['model_dir'] = (ou_df['model_total'] - ou_df['vegas_total']).apply(
    lambda x: 'over' if x > 0 else 'under')
over_games  = ou_df[ou_df['model_dir'] == 'over']
under_games = ou_df[ou_df['model_dir'] == 'under']
print(f'Model says OVER:  {len(over_games)} games, '
      f'{pct((over_games["total_ou_result"] == "over").sum(), len(over_games))} hit')
print(f'Model says UNDER: {len(under_games)} games, '
      f'{pct((under_games["total_ou_result"] == "under").sum(), len(under_games))} hit')

# ── 6. Quick summary ──────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('6. PICKS PERFORMANCE SUMMARY (current thresholds)')
print('=' * 60)
for tier_col, correct_col, actual_col, label in [
    ('ml_tier',    'ml_correct',    None,              'ML'),
    ('total_tier', 'model_total_dir','total_ou_result', 'Total'),
]:
    for tier in ['take', 'lean']:
        if label == 'ML':
            sub = df[(df[tier_col] == tier) & df['ml_correct'].isin([True, False])]
            hits = (sub['ml_correct'] == True).sum()
            n    = len(sub)
        else:
            sub = df[(df[tier_col] == tier) &
                     df[correct_col].isin(['over','under']) &
                     df[actual_col].isin(['over','under'])]
            hits = (sub[correct_col] == sub[actual_col]).sum()
            n    = len(sub)
        ev = hits/n * 100 if n else 0
        # EV at -110
        ev_110 = (hits * (100/110) - (n - hits)) if n else 0
        print(f'  {label} {tier:5s}: {pct(hits,n)}  EV at -110: {ev_110:+.0f}u on {n} bets')
