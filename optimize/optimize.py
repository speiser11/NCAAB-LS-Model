"""
TrueLine LS — Model Optimization v2
=====================================
Improvements over v1:
  1. Captures Torvik four factors (eFG%, OR%, TO%, FTR) — 8 extra features per game
  2. Season weighting (2025=3x, 2024=2x, 2023=1x) throughout all models
  3. Two model outputs:
       a) Weighted formula optimizer — best scoreScale + homeCourt for the app
       b) Ridge regression — learns optimal weights for all features (max accuracy)
  4. Out-of-sample validation: trains on 2022-24, tests on 2024-25
  5. Prints side-by-side comparison so you know exactly what you gain

Requirements:
  pip install requests pandas numpy tqdm scikit-learn

Usage:
  1. Paste your Odds API key into ODDS_API_KEY below
  2. python -X utf8 optimize.py
  3. First run fetches & caches all data (~20 min). Re-runs are instant.
"""

import json, math, re, sys, time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error
except ImportError:
    print('ERROR: scikit-learn not installed. Run: pip install scikit-learn')
    sys.exit(1)

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
ODDS_API_KEY = "b3e1ccd8a2f25665490e3c35abde718c"

DATA_DIR = Path(__file__).parent / "data"

SEASONS = {
    2023: ("20221101", "20230401"),
    2024: ("20231101", "20240401"),
    2025: ("20241101", "20250401"),
}

# Recent seasons matter more — affects both grid search and regression
SEASON_WEIGHTS = {2023: 1, 2024: 2, 2025: 3}

# Grid search space for formula optimizer
SCORE_SCALES = np.round(np.arange(0.90, 1.16, 0.02), 3).tolist()
HOME_COURTS  = np.round(np.arange(1.0,  5.5,  0.5),  1).tolist()

NCAAB_SIGMA  = 10.5

# ══════════════════════════════════════════════════════════
# TEAM NAME NORMALIZATION
# ══════════════════════════════════════════════════════════
ALIAS = {
    'connecticut': 'uconn', 'uconn huskies': 'uconn',
    'north carolina state': 'nc state', 'nc state wolfpack': 'nc state',
    'n.c. state': 'nc state',
    'southern california': 'usc', 'usc trojans': 'usc',
    'louisiana state': 'lsu', 'lsu tigers': 'lsu',
    'virginia commonwealth': 'vcu', 'vcu rams': 'vcu',
    'central florida': 'ucf', 'ucf knights': 'ucf',
    'southern methodist': 'smu', 'smu mustangs': 'smu',
    "texas a&m": 'texas am', "texas a&m aggies": 'texas am',
    'ole miss': 'mississippi', 'mississippi rebels': 'mississippi',
    'pittsburgh': 'pitt', 'pitt panthers': 'pitt',
    'miami (fl)': 'miami fl', 'miami florida': 'miami fl',
    'miami hurricanes': 'miami fl',
    'miami (oh)': 'miami oh', 'miami ohio': 'miami oh',
    'miami redhawks': 'miami oh',
    'penn': 'pennsylvania', 'penn quakers': 'pennsylvania',
    'umass': 'massachusetts', 'massachusetts minutemen': 'massachusetts',
    'cal': 'california', 'california golden bears': 'california',
    "hawai'i": 'hawaii', 'hawaii rainbow warriors': 'hawaii',
    "saint mary's (ca)": "saint mary's", "st. mary's (ca)": "saint mary's",
    "st mary's": "saint mary's",
    'byu': 'brigham young', 'byu cougars': 'brigham young',
    'unlv rebels': 'unlv', 'louisville cardinals': 'louisville',
    'unc wilmington': 'unc wilmington', 'uncw seahawks': 'unc wilmington',
    'ut martin': 'tennessee martin', 'utep miners': 'utep',
    'utsa roadrunners': 'utsa',
    'fiu': 'florida international',
    'florida international panthers': 'florida international',
    'fau': 'florida atlantic', 'florida atlantic owls': 'florida atlantic',
    'uab blazers': 'uab', 'uic flames': 'uic',
    'uta mavericks': 'ut arlington', 'ut arlington': 'ut arlington',
    'texas arlington': 'ut arlington',
    'umkc': 'kansas city', 'kansas city roos': 'kansas city',
    'siu edwardsville': 'siue', 'siue cougars': 'siue',
    'loyola (il)': 'loyola chicago',
    'loyola chicago ramblers': 'loyola chicago',
    "st. john's (ny)": "st. john's", "st john's red storm": "st. john's",
    'njit': 'njit', 'njit highlanders': 'njit',
    'tcu horned frogs': 'tcu',
    'ut rio grande valley': 'utrgv',
    'southeast missouri state': 'se missouri state',
    'long island university': 'liu', 'liu sharks': 'liu',
    'ipfw': 'purdue fort wayne',
    'purdue fort wayne mastodons': 'purdue fort wayne',
    'uc irvine anteaters': 'uc irvine',
    'uc santa barbara gauchos': 'uc santa barbara',
    'uc san diego tritons': 'uc san diego',
    'uc davis aggies': 'uc davis',
    'uc riverside highlanders': 'uc riverside',
    'gardner-webb': 'gardner webb',
    'gardner webb runnin bulldogs': 'gardner webb',
    "mount st. mary's": 'mount st. marys',
    'mount st. marys mountaineers': 'mount st. marys',
    "saint peter's": "saint peter's",
    "saint peter's peacocks": "saint peter's",
}

def norm_team(raw):
    if not raw:
        return ''
    s = raw.lower()
    s = re.sub(r"[\u2018\u2019''`]", "'", s)
    s = re.sub(r'\s*\([^)]{0,6}\)\s*', ' ', s)
    s = re.sub(r"[^a-z0-9\s'&.-]", '', s)
    s = re.sub(r'([a-z])\.([a-z])\.', r'\1\2', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r' st\.$', ' state', s)
    s = re.sub(r'\bst\b(?!\.)', 'state', s)
    return ALIAS.get(s, s)

def match_team(full_name, ratings_keys_set, ratings_keys_list):
    norm = norm_team(full_name)
    if norm in ratings_keys_set:
        return norm
    words = norm.split()
    for length in range(len(words) - 1, 0, -1):
        shorter = ' '.join(words[:length])
        if shorter in ratings_keys_set:
            return shorter
    nw = [w for w in words if len(w) > 2]
    if not nw:
        return None
    best_score, best_key = 0.0, None
    for key in ratings_keys_list:
        kw = [w for w in key.split() if len(w) > 2]
        if not kw:
            continue
        shared = len(set(nw) & set(kw))
        if not shared:
            continue
        score = (2 * shared) / (len(nw) + len(kw))
        if score > best_score and score >= 0.5:
            best_score, best_key = score, key
    return best_key

# ══════════════════════════════════════════════════════════
# DATE UTILITIES
# ══════════════════════════════════════════════════════════
def date_range(start_str, end_str):
    d   = date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:]))
    end = date(int(end_str[:4]),   int(end_str[4:6]),   int(end_str[6:]))
    while d <= end:
        yield d.strftime('%Y%m%d')
        d += timedelta(days=1)

def week_anchor(date_str):
    d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
    monday = d - timedelta(days=d.weekday())
    return monday.strftime('%Y%m%d')

def torvik_season_year(date_str):
    y, m = int(date_str[:4]), int(date_str[4:6])
    return y + 1 if m >= 10 else y

# ══════════════════════════════════════════════════════════
# TORVIK FETCHER — captures adjO/adjD/adjT + four factors
#
# Torvik trank.php JSON field mapping (confirmed empirically):
#   [0]  team name       [1]  adjO (off efficiency)
#   [2]  adjD (def eff)  [3]  barthag (win prob vs avg)
#   [4]  record string   [5]  wins     [6]  games
#   [7]  off_eFG%        [8]  def_eFG%
#   [9]  off_OR%         [10] def_OR%
#   [11] off_TO%         [12] def_TO%
#   [13] off_FTR         [14] def_FTR
#   [15] adjT (tempo)
# ══════════════════════════════════════════════════════════
TORVIK_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
_torvik_session = None

def get_torvik_session():
    global _torvik_session
    if _torvik_session is not None:
        return _torvik_session
    s = requests.Session()
    s.headers.update(TORVIK_HEADERS)
    try:
        s.post('https://barttorvik.com/trank.php?json=1',
               data={'js_test_submitted': '1'}, timeout=15)
    except Exception:
        pass
    _torvik_session = s
    return s

def fetch_torvik(date_str):
    """Fetch Torvik ratings + four factors + games played for date_str. Cached to disk.
    Automatically invalidates old caches missing four factor or games fields."""
    cache = DATA_DIR / 'torvik' / f'{date_str}.json'
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data:
                sample = next(iter(data.values()))
                if 'off_eFG' in sample and 'games' in sample:
                    return data
            # Cache is stale — delete and re-fetch
            cache.unlink()
        except Exception:
            cache.unlink()

    year = torvik_season_year(date_str)
    url  = f'https://barttorvik.com/trank.php?year={year}&dte={date_str}&json=1'
    try:
        r = get_torvik_session().get(url, timeout=20)
        r.raise_for_status()
        raw_data = r.json()
    except Exception as e:
        print(f'  Torvik {date_str}: {e}')
        return {}

    out = {}
    for row in raw_data:
        if not isinstance(row, list) or len(row) < 16:
            continue
        try:
            key = norm_team(str(row[0]))
            out[key] = {
                'raw':     str(row[0]),
                'adjO':    float(row[1]),
                'adjD':    float(row[2]),
                'barthag': float(row[3]),
                'games':   int(row[6]),    # games played at this weekly snapshot
                'adjT':    float(row[15]),
                # Four factors
                'off_eFG': float(row[7]),
                'def_eFG': float(row[8]),
                'off_OR':  float(row[9]),
                'def_OR':  float(row[10]),
                'off_TO':  float(row[11]),
                'def_TO':  float(row[12]),
                'off_FTR': float(row[13]),
                'def_FTR': float(row[14]),
            }
        except (ValueError, TypeError):
            continue

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    time.sleep(0.4)
    return out

# ══════════════════════════════════════════════════════════
# ESPN FETCHER
# ══════════════════════════════════════════════════════════
ESPN_URL = ('https://site.api.espn.com/apis/site/v2/sports/basketball'
            '/mens-college-basketball/scoreboard')

def fetch_espn(date_str):
    cache = DATA_DIR / 'espn' / f'{date_str}.json'
    if cache.exists():
        return json.loads(cache.read_text())

    try:
        r = requests.get(ESPN_URL, params={'dates': date_str, 'groups': 50,
                                           'limit': 300}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'  ESPN {date_str}: {e}')
        return []

    games = []
    for ev in data.get('events', []):
        comp   = ev['competitions'][0]
        status = comp['status']['type']['name']
        if status != 'STATUS_FINAL':
            continue
        try:
            home = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
            away = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
            games.append({
                'home':       home['team']['displayName'],
                'away':       away['team']['displayName'],
                'home_score': int(home['score']),
                'away_score': int(away['score']),
                'neutral':    comp.get('neutralSite', False),
            })
        except (KeyError, StopIteration, ValueError):
            continue

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(games))
    return games

# ══════════════════════════════════════════════════════════
# THE ODDS API HISTORICAL FETCHER
# ══════════════════════════════════════════════════════════
ODDS_HIST_URL = 'https://api.the-odds-api.com/v4/historical/sports/basketball_ncaab/odds/'

def fetch_odds_historical(date_str):
    cache = DATA_DIR / 'odds' / f'{date_str}.json'
    if cache.exists():
        return json.loads(cache.read_text())

    iso  = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    ts   = f'{iso}T17:00:00Z'
    try:
        r = requests.get(ODDS_HIST_URL, params={
            'apiKey':     ODDS_API_KEY,
            'regions':    'us',
            'markets':    'spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
            'date':       ts,
        }, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f'  Odds API {date_str}: {e}')
        return {}

    games_raw = payload.get('data', payload) if isinstance(payload, dict) else payload
    result = {}
    for game in (games_raw or []):
        home_raw = game.get('home_team', '')
        away_raw = game.get('away_team', '')
        spread, total = None, None
        for bm in game.get('bookmakers', []):
            for mkt in bm.get('markets', []):
                if mkt['key'] == 'spreads' and spread is None:
                    for oc in mkt.get('outcomes', []):
                        if norm_team(oc['name']) == norm_team(home_raw):
                            spread = float(oc['point'])
                            break
                elif mkt['key'] == 'totals' and total is None:
                    for oc in mkt.get('outcomes', []):
                        if oc['name'].lower() == 'over':
                            total = float(oc['point'])
                            break
            if spread is not None and total is not None:
                break
        key = '|'.join(sorted([norm_team(home_raw), norm_team(away_raw)]))
        result[key] = {'spread': spread, 'total': total, 'home': home_raw, 'away': away_raw}

    remaining = r.headers.get('x-requests-remaining', '?')
    used      = r.headers.get('x-requests-used', '?')
    print(f'  Odds {date_str}: {len(result)} games | credits used={used} remaining={remaining}')

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    time.sleep(0.25)
    return result

# ══════════════════════════════════════════════════════════
# BUILD DATASET — expanded with four factors + season column
# ══════════════════════════════════════════════════════════
def build_dataset(seasons=None):
    if seasons is None:
        seasons = list(SEASONS.keys())

    all_dates = []
    for yr in seasons:
        start, end = SEASONS[yr]
        all_dates.extend(date_range(start, end))
    seen = set()
    all_dates = [d for d in all_dates if not (d in seen or seen.add(d))]

    anchors = sorted(set(week_anchor(d) for d in all_dates))
    print(f'Pre-fetching Torvik ratings for {len(anchors)} weekly snapshots...')
    torvik_by_anchor = {}
    for anc in tqdm(anchors, desc='Torvik'):
        torvik_by_anchor[anc] = fetch_torvik(anc)

    rows = []
    match_misses = 0

    print(f'\nProcessing {len(all_dates)} dates...')
    for date_str in tqdm(all_dates, desc='Dates'):
        anc     = week_anchor(date_str)
        ratings = torvik_by_anchor.get(anc, {})
        if not ratings:
            continue

        r_keys_set  = set(ratings.keys())
        r_keys_list = list(ratings.keys())

        espn_games = fetch_espn(date_str)
        if not espn_games:
            continue

        odds = fetch_odds_historical(date_str)

        for g in espn_games:
            h_key = match_team(g['home'], r_keys_set, r_keys_list)
            a_key = match_team(g['away'], r_keys_set, r_keys_list)
            if not h_key or not a_key:
                match_misses += 1
                continue

            home_r = ratings[h_key]
            away_r = ratings[a_key]

            actual_margin = g['home_score'] - g['away_score']
            actual_total  = g['home_score'] + g['away_score']

            game_key  = '|'.join(sorted([norm_team(g['home']), norm_team(g['away'])]))
            game_odds = odds.get(game_key)
            if not game_odds:
                for ok, ov in odds.items():
                    if match_team(ov['home'], r_keys_set, r_keys_list) in (h_key, a_key):
                        game_odds = ov
                        break

            rows.append({
                'date':          date_str,
                'season':        torvik_season_year(date_str),
                'home':          g['home'],
                'away':          g['away'],
                'home_score':    g['home_score'],
                'away_score':    g['away_score'],
                'actual_margin': actual_margin,
                'actual_total':  actual_total,
                'neutral':       g['neutral'],
                # Core efficiency
                'h_adjO':    home_r['adjO'],   'h_adjD':    home_r['adjD'],
                'h_adjT':    home_r['adjT'],   'h_barthag': home_r['barthag'],
                'h_games':   home_r.get('games', 20),
                'a_adjO':    away_r['adjO'],   'a_adjD':    away_r['adjD'],
                'a_adjT':    away_r['adjT'],   'a_barthag': away_r['barthag'],
                'a_games':   away_r.get('games', 20),
                # Four factors — home
                'h_off_eFG': home_r['off_eFG'], 'h_def_eFG': home_r['def_eFG'],
                'h_off_OR':  home_r['off_OR'],  'h_def_OR':  home_r['def_OR'],
                'h_off_TO':  home_r['off_TO'],  'h_def_TO':  home_r['def_TO'],
                'h_off_FTR': home_r['off_FTR'], 'h_def_FTR': home_r['def_FTR'],
                # Four factors — away
                'a_off_eFG': away_r['off_eFG'], 'a_def_eFG': away_r['def_eFG'],
                'a_off_OR':  away_r['off_OR'],  'a_def_OR':  away_r['def_OR'],
                'a_off_TO':  away_r['off_TO'],  'a_def_TO':  away_r['def_TO'],
                'a_off_FTR': away_r['off_FTR'], 'a_def_FTR': away_r['def_FTR'],
                # Vegas lines
                'vegas_spread': game_odds['spread'] if game_odds else None,
                'vegas_total':  game_odds['total']  if game_odds else None,
            })

    df = pd.DataFrame(rows)
    dataset_path = DATA_DIR / 'dataset.csv'
    df.to_csv(dataset_path, index=False)

    n_spread = df['vegas_spread'].notna().sum()
    print(f'\nDataset: {len(df):,} games | {n_spread:,} with spreads')
    print(f'Team match misses: {match_misses}')
    print(f'Saved to {dataset_path}')
    return df

# ══════════════════════════════════════════════════════════
# GAME WEIGHTS  (season recency × games-played reliability)
# ══════════════════════════════════════════════════════════
def get_sample_weights(df):
    """
    Combined weight = season_weight × early_season_weight.

    Season weight:       2025=3x  2024=2x  2023=1x  (recent seasons matter more)
    Early-season weight: scales by weeks into the season. Torvik ratings are noisy
                         for the first 6 weeks (Nov 1 → ~Dec 10) because teams have
                         played only a handful of games. After ~8 weeks ratings
                         stabilize. This directly targets the Nov/Dec MAE=11.0 vs
                         Jan+ MAE=8.5 gap we measured in the data.

    Note: Torvik's 'games' field carries over from the prior season so we can't
    use it directly. We compute season progress from the game date instead.
    """
    season_w = df['season'].map(SEASON_WEIGHTS).fillna(1).values

    # Compute weeks into the current season (season starts ~Nov 1)
    dates = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    # Season year: year the season ends (e.g. 2025 = 2024-25 season)
    season_yr = df['season'].values
    # Nov 1 of the start year (e.g. 2024-11-01 for the 2024-25 season)
    season_start = pd.to_datetime(
        pd.Series(season_yr - 1).astype(str) + '-11-01'
    ).values
    days_in  = (dates.values - season_start).astype('timedelta64[D]').astype(float)
    weeks_in = days_in / 7.0

    # Full weight at 8 weeks (≈ end of December); floor at 0.15 so Nov games
    # are down-weighted but not completely ignored
    early_w = np.clip(weeks_in / 8.0, 0.15, 1.0)

    return season_w * early_w

# ══════════════════════════════════════════════════════════
# FORMULA SCORING — vectorized, supports sample weights
# ══════════════════════════════════════════════════════════
def score_params(df, score_scale, home_court, weights=None):
    poss    = np.sqrt(df['h_adjT'] * df['a_adjT'])
    # Normalize adjD against D1 average so scoreScale ≈ 1.0 rather than compensating
    # for Torvik adjO/adjD averaging ~106/105 instead of 100/100
    h_score = df['h_adjO'] * (df['a_adjD'] / _ADJ_D_MEAN) * poss / 100 * score_scale
    a_score = df['a_adjO'] * (df['h_adjD'] / _ADJ_D_MEAN) * poss / 100 * score_scale
    hc_arr  = np.where(df['neutral'].astype(bool), 0.0, home_court)
    margin  = (h_score - a_score) + hc_arr
    total   = h_score + a_score

    error   = np.abs(df['actual_margin'] - margin)

    if weights is not None:
        w = weights / weights.sum()
        mae      = (error * w).sum()
        within35 = ((error <= 3.5) * w).sum()
        correct  = (((margin > 0) == (df['actual_margin'] > 0)).astype(float) * w).sum()
    else:
        mae      = error.mean()
        within35 = (error <= 3.5).mean()
        correct  = ((margin > 0) == (df['actual_margin'] > 0)).mean()

    # ATS (unweighted — betting edge is binary)
    ats_mask = df['vegas_spread'].notna()
    ats_df   = df[ats_mask].copy()
    ats_df['_margin'] = margin[ats_mask].values
    ats_df['_cover']  = ats_df['actual_margin'] + ats_df['vegas_spread']
    ats_df['_backed'] = ats_df['_margin'] >= 0
    ats_df = ats_df[ats_df['_cover'] != 0]

    ats_correct = (ats_df['_backed'] == (ats_df['_cover'] > 0))
    ats_rate    = ats_correct.mean() if len(ats_df) else float('nan')

    ats_df['_edge'] = np.abs(ats_df['_margin'] + ats_df['vegas_spread'])
    hi_edge = ats_df[ats_df['_edge'] >= 3.0]
    hi_ats  = (hi_edge['_backed'] == (hi_edge['_cover'] > 0)).mean() if len(hi_edge) else float('nan')

    return {
        'mae': mae, 'within35': within35, 'correct': correct,
        'ats': ats_rate, 'hi_ats': hi_ats,
        'n_games': len(df), 'n_ats': int(ats_mask.sum()),
    }

# ══════════════════════════════════════════════════════════
# WEIGHTED FORMULA OPTIMIZATION (grid search)
# ══════════════════════════════════════════════════════════
def run_formula_optimization(df):
    weights = get_sample_weights(df)

    print('\n' + '═' * 64)
    print('WEIGHTED FORMULA OPTIMIZER  (scoreScale × homeCourt)')
    print(f'Season weights: 2023=1x  2024=2x  2025=3x')
    print(f'Testing {len(SCORE_SCALES)} × {len(HOME_COURTS)} = '
          f'{len(SCORE_SCALES)*len(HOME_COURTS)} combinations')
    print('═' * 64)

    results = []
    for ss in tqdm(SCORE_SCALES, desc='scoreScale'):
        for hc in HOME_COURTS:
            m = score_params(df, ss, hc, weights=weights)
            results.append({'score_scale': ss, 'home_court': hc, **m})

    res_df = pd.DataFrame(results)
    res_df.to_csv(DATA_DIR / 'formula_optimization.csv', index=False)

    def pct(v):
        return f'{v*100:.1f}%' if not math.isnan(v) else '—'

    print('\n── Top 5 by weighted MAE ───────────────────────────────────')
    for _, r in res_df.nsmallest(5, 'mae').iterrows():
        print(f'  scoreScale={r.score_scale:.2f}  homeCourt={r.home_court:.1f}  '
              f'wMAE={r.mae:.3f}  correct={pct(r.correct)}  ATS={pct(r.ats)}')

    best = res_df.loc[res_df['mae'].idxmin()]
    print(f'\n  → Best formula params:  scoreScale={best.score_scale:.2f}  '
          f'homeCourt={best.home_court:.1f}  wMAE={best.mae:.3f}')

    return best

# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR REGRESSION
#
# All deltas constructed so that positive value = home team advantage.
# Each feature should have a positive regression coefficient.
# ══════════════════════════════════════════════════════════
FEATURE_NAMES = [
    'delta_adjO',     # home.adjO - away.adjO  (better off = higher margin)
    'delta_adjD',     # away.adjD - home.adjD  (lower adjD = better def → reversed)
    'delta_eFG',      # home.off_eFG - away.off_eFG  (shoot better = win more)
    'delta_def_eFG',  # away.def_eFG - home.def_eFG  (lower = better def → reversed)
    'delta_OR',       # home.off_OR - away.off_OR  (offensive rebounding edge)
    'delta_TO',       # away.off_TO - home.off_TO  (force more TOs = win more → reversed)
    'delta_FTR',      # home.off_FTR - away.off_FTR  (get to line more = win more)
    'home_court',     # 1 if not neutral, 0 if neutral  (coefficient = HCA in pts)
    # Removed: avg_adjT (r=+0.03 with margin, nearly zero signal)
    # Removed: delta_barthag (derived from adjO+adjD, causes negative coefficient / double-count)
]

# Mean values of non-delta features used for centering
_ADJ_O_MEAN = 106.16   # approximate D1 average adjO
_ADJ_D_MEAN = 104.82   # approximate D1 average adjD

def build_features(df):
    return pd.DataFrame({
        # Center adjO/adjD around D1 average so intercept stays near 0
        'delta_adjO':    df['h_adjO']    - df['a_adjO'],
        'delta_adjD':    df['a_adjD']    - df['h_adjD'],
        'delta_eFG':     df['h_off_eFG'] - df['a_off_eFG'],
        'delta_def_eFG': df['a_def_eFG'] - df['h_def_eFG'],
        'delta_OR':      df['h_off_OR']  - df['a_off_OR'],
        'delta_TO':      df['a_off_TO']  - df['h_off_TO'],
        'delta_FTR':     df['h_off_FTR'] - df['a_off_FTR'],
        'home_court':    (~df['neutral'].astype(bool)).astype(float),
    }, columns=FEATURE_NAMES)

# ══════════════════════════════════════════════════════════
# RIDGE REGRESSION MODEL
# ══════════════════════════════════════════════════════════
def run_regression_model(df):
    print('\n' + '═' * 64)
    print('RIDGE REGRESSION MODEL  (all features + season weights)')
    print('Out-of-sample validation: train 2022-24  →  test 2024-25')
    print('═' * 64)

    if 'h_off_eFG' not in df.columns:
        print('  Missing four-factor columns. Delete dataset.csv and re-run.')
        return None

    X = build_features(df)
    y = df['actual_margin'].values
    weights = get_sample_weights(df)

    train_mask = df['season'] < 2025
    test_mask  = df['season'] == 2025

    X_train = X[train_mask].values
    y_train = y[train_mask]
    w_train = weights[train_mask]
    X_test  = X[test_mask].values
    y_test  = y[test_mask]

    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    # Show effective weight breakdown so user can see games-weighting in action
    w_all   = get_sample_weights(df)
    early   = df['season'].eq(2025) & df['date'].astype(str).str[4:6].isin(['11','12'])
    late    = df['season'].eq(2025) & ~df['date'].astype(str).str[4:6].isin(['11','12'])
    print(f'\nTrain: {n_train:,} games (2022-24) | Test: {n_test:,} games (2024-25)')
    avg_early = w_all[early.values].mean() if early.any() else 0
    avg_late  = w_all[late.values].mean()  if late.any()  else 0
    print(f'Avg weight — early season (Nov/Dec): {avg_early:.2f} | late season (Jan+): {avg_late:.2f}')

    # RidgeCV auto-selects regularization via leave-one-out CV
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    model  = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X_train, y_train, sample_weight=w_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    def metrics(y_true, y_pred):
        err = np.abs(y_true - y_pred)
        return {
            'mae':      err.mean(),
            'within35': (err <= 3.5).mean(),
            'within7':  (err <= 7.0).mean(),
            'correct':  ((y_pred > 0) == (y_true > 0)).mean(),
        }

    tr = metrics(y_train, pred_train)
    te = metrics(y_test,  pred_test)

    print(f'\n  {"Metric":<20} {"Train (2022-24)":>16} {"Test (2024-25)":>16}')
    print(f'  {"─"*20} {"─"*16} {"─"*16}')
    print(f'  {"MAE":<20} {tr["mae"]:>15.2f}  {te["mae"]:>15.2f}')
    print(f'  {"Correct side":<20} {tr["correct"]*100:>14.1f}%  {te["correct"]*100:>14.1f}%')
    print(f'  {"Within 3.5 pts":<20} {tr["within35"]*100:>14.1f}%  {te["within35"]*100:>14.1f}%')
    print(f'  {"Within 7 pts":<20} {tr["within7"]*100:>14.1f}%  {te["within7"]*100:>14.1f}%')

    print(f'\n  Regularization alpha selected: {model.alpha_}')
    print(f'\n  Learned coefficients:')
    print(f'  {"Feature":<20} {"Coeff":>10}  Note')
    print(f'  {"─"*20} {"─"*10}  {"─"*35}')
    for name, coef in zip(FEATURE_NAMES, model.coef_):
        note = ''
        if name == 'home_court':
            note = f'← implied HCA = {coef:.2f} pts'
        elif name == 'avg_adjT':
            note = f'← pace effect per possession-unit'
        print(f'  {name:<20} {coef:>10.4f}  {note}')
    if abs(model.intercept_) > 0.1:
        print(f'  {"intercept":<20} {model.intercept_:>10.4f}  ← systematic bias (ideally ~0)')

    return model, te

# ══════════════════════════════════════════════════════════
# BASELINE DIAGNOSTICS (current app settings)
# ══════════════════════════════════════════════════════════
def run_baseline(df):
    print('═' * 64)
    print('BASELINE  (current app: scoreScale=0.96, homeCourt=3.5)')
    print('═' * 64)

    m = score_params(df, 0.96, 3.5)

    print(f'  Games:        {m["n_games"]:,}')
    print(f'  MAE:          {m["mae"]:.2f} pts')
    print(f'  Correct side: {m["correct"]*100:.1f}%')
    print(f'  Within 3.5:   {m["within35"]*100:.1f}%')
    print(f'  ATS:          {m["ats"]*100:.1f}%  ({m["n_ats"]:,} games)')
    print(f'  ATS ≥3 edge:  {m["hi_ats"]*100:.1f}%')
    return m

# ══════════════════════════════════════════════════════════
# SUMMARY + INDEX.HTML UPDATE INSTRUCTIONS
# ══════════════════════════════════════════════════════════
def print_summary(baseline, best_formula, regression_result):
    print('\n' + '═' * 64)
    print('SUMMARY — WHAT EACH MODEL GIVES YOU')
    print('═' * 64)

    reg_model, reg_te = regression_result if regression_result else (None, None)

    print(f'\n  {"Model":<30} {"MAE":>8}  {"Correct":>9}  {"Within3.5":>10}')
    print(f'  {"─"*30} {"─"*8}  {"─"*9}  {"─"*10}')
    print(f'  {"Current app (1.12 / 2.0)":<30} {baseline["mae"]:>8.2f}  '
          f'{baseline["correct"]*100:>8.1f}%  {baseline["within35"]*100:>9.1f}%')

    # Re-score with best formula params
    bf_m = score_params(df_global,
                        float(best_formula['score_scale']),
                        float(best_formula['home_court']))
    print(f'  {"Weighted formula opt":<30} {bf_m["mae"]:>8.2f}  '
          f'{bf_m["correct"]*100:>8.1f}%  {bf_m["within35"]*100:>9.1f}%')

    if reg_te:
        print(f'  {"Ridge regression (OOS 2024-25)":<30} {reg_te["mae"]:>8.2f}  '
              f'{reg_te["correct"]*100:>8.1f}%  {reg_te["within35"]*100:>9.1f}%')

    print(f'\n  Breakeven ATS: 52.4%')

    print(f'\n  To update the app with weighted formula params:')
    print(f'    scoreScale = {best_formula["score_scale"]:.2f}')
    print(f'    homeCourt  = {best_formula["home_court"]:.1f}')

    if reg_model is not None:
        hc_idx = FEATURE_NAMES.index('home_court')
        hc_val = reg_model.coef_[hc_idx]
        print(f'\n  Regression model implies home court = {hc_val:.2f} pts')
        print(f'  (vs {best_formula["home_court"]:.1f} pts from formula optimizer)')
        print(f'\n  Full regression results saved → data/regression_coefficients.json')

        coeff_path = DATA_DIR / 'regression_coefficients.json'
        coeff_path.write_text(json.dumps({
            'features':    FEATURE_NAMES,
            'coefficients': model_coef_list(reg_model),
            'intercept':   reg_model.intercept_,
            'alpha':       reg_model.alpha_,
            'oos_mae':     reg_te['mae'],
            'oos_correct': reg_te['correct'],
        }, indent=2))

    print('═' * 64)

def model_coef_list(model):
    return [round(float(c), 6) for c in model.coef_]

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
df_global = None  # set in main so summary can access it

if __name__ == '__main__':
    print('TrueLine LS — Model Optimization v2')
    print('=' * 64)

    if ODDS_API_KEY == 'YOUR_KEY_HERE':
        print('ERROR: Set ODDS_API_KEY at the top of this script.')
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load or build dataset
    dataset_path = DATA_DIR / 'dataset.csv'
    needs_rebuild = True
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        if 'h_off_eFG' in df.columns and 'season' in df.columns:
            needs_rebuild = False
            print(f'Loaded cached dataset: {len(df):,} games '
                  f'({df["vegas_spread"].notna().sum():,} with spreads)')
        else:
            print('Dataset missing new columns — rebuilding with four factors...')

    if needs_rebuild:
        print('Building dataset (fetching from APIs)...')
        df = build_dataset()

    if df is None or df.empty:
        print('No data. Check API key and internet connection.')
        sys.exit(1)

    df_global = df

    # 1. Baseline
    print()
    baseline = run_baseline(df)

    # 2. Weighted formula optimization
    best_formula = run_formula_optimization(df)

    # 3. Ridge regression with all features
    regression_result = run_regression_model(df)

    # 4. Summary
    print_summary(baseline, best_formula, regression_result)
