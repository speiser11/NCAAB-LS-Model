"""
TrueLine LS — Model Optimization
=================================
Pulls historical ESPN scores, point-in-time Torvik ratings, and The Odds API
historical lines. Grid-searches over key model parameters to find optimal values.

Setup:
  pip install requests pandas numpy tqdm

Usage:
  1. Paste your Odds API key into ODDS_API_KEY below
  2. python optimize.py
  3. On first run it fetches & caches all data (slow, ~10-20 min)
  4. Re-runs are instant — everything cached locally in ./data/

The Odds API historical calls use credits. Each date = 1 call.
~150 dates/season × 3 seasons = ~450 calls total. Run once, cache forever.
"""

import json, math, re, sys, time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

# ══════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════
ODDS_API_KEY = "YOUR_KEY_HERE"   # paste your The Odds API key

DATA_DIR = Path(__file__).parent / "data"

# Season year → (start YYYYMMDD, end YYYYMMDD)
# Season year 2025 = the 2024-25 season (Torvik's "year" param)
SEASONS = {
    2023: ("20221101", "20230401"),
    2024: ("20231101", "20240401"),
    2025: ("20241101", "20250401"),
}

# Grid search space — adjust to widen or narrow
SCORE_SCALES = np.round(np.arange(0.96, 1.14, 0.02), 3).tolist()
HOME_COURTS  = np.round(np.arange(2.0,  5.5,  0.5),  1).tolist()

NCAAB_SIGMA  = 10.5   # historical std dev of NCAAB game margins

# ══════════════════════════════════════════════════════════
# TEAM NAME NORMALIZATION  (faithful Python port of app logic)
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
    s = re.sub(r'\s*\([^)]{0,6}\)\s*', ' ', s)    # strip (NY), (FL), etc.
    s = re.sub(r"[^a-z0-9\s'&.-]", '', s)
    s = re.sub(r'([a-z])\.([a-z])\.', r'\1\2', s)  # n.c. → nc
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r' st\.$', ' state', s)              # iowa st. → iowa state
    s = re.sub(r'\bst\b(?!\.)', 'state', s)         # standalone st → state
    return ALIAS.get(s, s)

def match_team(full_name, ratings_keys_set, ratings_keys_list):
    """Port of JS matchTeam — tries direct, progressive strip, then Dice."""
    norm = norm_team(full_name)
    if norm in ratings_keys_set:
        return norm

    # Strip trailing words one at a time (removes mascot)
    words = norm.split()
    for length in range(len(words) - 1, 0, -1):
        shorter = ' '.join(words[:length])
        if shorter in ratings_keys_set:
            return shorter

    # Dice coefficient on shared words (>2 chars)
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
    """Yield YYYYMMDD strings from start_str to end_str inclusive."""
    d   = date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:]))
    end = date(int(end_str[:4]),   int(end_str[4:6]),   int(end_str[6:]))
    while d <= end:
        yield d.strftime('%Y%m%d')
        d += timedelta(days=1)

def week_anchor(date_str):
    """Return YYYYMMDD of the Monday of the week containing date_str."""
    d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
    monday = d - timedelta(days=d.weekday())
    return monday.strftime('%Y%m%d')

def torvik_season_year(date_str):
    y, m = int(date_str[:4]), int(date_str[4:6])
    return y + 1 if m >= 10 else y

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
# TORVIK FETCHER  (weekly snapshots = ~65 calls/season)
# ══════════════════════════════════════════════════════════
TORVIK_URL = 'https://barttorvik.com/trank.php?json=1'

def fetch_torvik(date_str):
    """Fetch Torvik ratings as-of date_str. Cached to disk."""
    cache = DATA_DIR / 'torvik' / f'{date_str}.json'
    if cache.exists():
        return json.loads(cache.read_text())

    year = torvik_season_year(date_str)
    url  = f'https://barttorvik.com/trank.php?year={year}&dte={date_str}&json=1'
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'  Torvik {date_str}: {e}')
        return {}

    out = {}
    for row in data:
        if not isinstance(row, list) or len(row) < 5:
            continue
        try:
            key = norm_team(str(row[0]))
            out[key] = {
                'raw':     str(row[0]),
                'adjO':    float(row[2]),
                'adjD':    float(row[3]),
                'adjT':    float(row[4]),
                'barthag': float(row[5]) if len(row) > 5 else 0.5,
            }
        except (ValueError, TypeError):
            continue

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    time.sleep(0.4)   # polite rate limit
    return out

# ══════════════════════════════════════════════════════════
# THE ODDS API HISTORICAL FETCHER
# ══════════════════════════════════════════════════════════
ODDS_HIST_URL = 'https://api.the-odds-api.com/v4/historical/sports/basketball_ncaab/odds/'

def fetch_odds_historical(date_str):
    """
    Snapshot at 17:00 UTC (noon ET) — captures pre-game lines for most NCAAB games.
    Returns {norm_key: {spread, total}} where norm_key = sorted normed team names.
    """
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

    # Response may be {"data": [...]} or just [...]
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
        result[key] = {
            'spread': spread, 'total': total,
            'home': home_raw, 'away': away_raw,
        }

    # Check remaining credits
    remaining = r.headers.get('x-requests-remaining', '?')
    used      = r.headers.get('x-requests-used', '?')
    print(f'  Odds {date_str}: {len(result)} games | credits used={used} remaining={remaining}')

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    time.sleep(0.25)
    return result

# ══════════════════════════════════════════════════════════
# PREDICT  (Python port of app logic)
# ══════════════════════════════════════════════════════════
def predict(home_r, away_r, neutral=False, home_court=3.5, score_scale=1.06):
    if not home_r or not away_r:
        return None
    poss    = math.sqrt(home_r['adjT'] * away_r['adjT'])
    h_score = (home_r['adjO'] * away_r['adjD'] / 100) * poss / 100 * score_scale
    a_score = (away_r['adjO'] * home_r['adjD'] / 100) * poss / 100 * score_scale
    hc      = 0.0 if neutral else home_court
    margin  = (h_score - a_score) + hc
    return {'margin': margin, 'total': h_score + a_score}

# ══════════════════════════════════════════════════════════
# BUILD DATASET
# ══════════════════════════════════════════════════════════
def build_dataset(seasons=None):
    """
    Fetches all historical data and returns a DataFrame.
    Torvik is fetched weekly (Monday anchors) to limit API calls.
    Everything is cached — safe to re-run without burning credits.
    """
    if seasons is None:
        seasons = list(SEASONS.keys())

    all_dates = []
    for yr in seasons:
        start, end = SEASONS[yr]
        all_dates.extend(date_range(start, end))
    # deduplicate
    seen = set()
    all_dates = [d for d in all_dates if not (d in seen or seen.add(d))]

    # Pre-fetch Torvik for all weekly anchors
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
            pred   = predict(home_r, away_r, g['neutral'])
            if not pred:
                continue

            actual_margin = g['home_score'] - g['away_score']
            actual_total  = g['home_score'] + g['away_score']

            # Match odds — try exact key first, then reversed, then fuzzy
            game_key = '|'.join(sorted([norm_team(g['home']), norm_team(g['away'])]))
            game_odds = odds.get(game_key)
            if not game_odds:
                # Try matching by individual team norm keys
                for ok, ov in odds.items():
                    parts = ok.split('|')
                    if norm_team(ov['home']) in r_keys_set and match_team(ov['home'], r_keys_set, r_keys_list) in (h_key, a_key):
                        game_odds = ov
                        break

            rows.append({
                'date':          date_str,
                'home':          g['home'],
                'away':          g['away'],
                'home_score':    g['home_score'],
                'away_score':    g['away_score'],
                'actual_margin': actual_margin,
                'actual_total':  actual_total,
                'neutral':       g['neutral'],
                # Store raw ratings so we can re-score without re-fetching
                'h_adjO':        home_r['adjO'],
                'h_adjD':        home_r['adjD'],
                'h_adjT':        home_r['adjT'],
                'a_adjO':        away_r['adjO'],
                'a_adjD':        away_r['adjD'],
                'a_adjT':        away_r['adjT'],
                # Default prediction (scoreScale=1.06, homeCourt=3.5)
                'pred_margin':   pred['margin'],
                'pred_total':    pred['total'],
                # Vegas lines (None if not available)
                'vegas_spread':  game_odds['spread'] if game_odds else None,
                'vegas_total':   game_odds['total']  if game_odds else None,
            })

    df = pd.DataFrame(rows)
    dataset_path = DATA_DIR / 'dataset.csv'
    df.to_csv(dataset_path, index=False)

    n_spread = df['vegas_spread'].notna().sum()
    n_total  = df['vegas_total'].notna().sum()
    print(f'\nDataset: {len(df):,} games | {n_spread:,} with spreads | {n_total:,} with totals')
    print(f'Team match misses: {match_misses}')
    print(f'Saved to {dataset_path}')
    return df

# ══════════════════════════════════════════════════════════
# SCORING — re-run predictions from stored raw ratings
# ══════════════════════════════════════════════════════════
def score_params(df, score_scale, home_court):
    """
    Vectorized re-score of all games in df with given params.
    Returns metrics dict.
    """
    poss     = np.sqrt(df['h_adjT'] * df['a_adjT'])
    h_score  = (df['h_adjO'] * df['a_adjD'] / 100) * poss / 100 * score_scale
    a_score  = (df['a_adjO'] * df['h_adjD'] / 100) * poss / 100 * score_scale
    hc_arr   = np.where(df['neutral'].astype(bool), 0.0, home_court)
    margin   = (h_score - a_score) + hc_arr
    total    = h_score + a_score

    error    = np.abs(df['actual_margin'] - margin)
    mae      = error.mean()
    within35 = (error <= 3.5).mean()
    within7  = (error <= 7.0).mean()
    correct  = ((margin > 0) == (df['actual_margin'] > 0)).mean()

    # ATS
    ats_mask = df['vegas_spread'].notna()
    ats_df   = df[ats_mask].copy()
    ats_df['_margin'] = margin[ats_mask].values
    ats_df['_cover']  = ats_df['actual_margin'] + ats_df['vegas_spread']
    ats_df['_backed'] = ats_df['_margin'] >= 0
    ats_df = ats_df[ats_df['_cover'] != 0]  # exclude pushes

    ats_correct = (ats_df['_backed'] == (ats_df['_cover'] > 0))
    ats_rate    = ats_correct.mean() if len(ats_df) else float('nan')

    # ATS by edge bucket
    ats_df['_edge'] = np.abs(ats_df['_margin'] + ats_df['vegas_spread'])
    hi_edge = ats_df[ats_df['_edge'] >= 3.0]
    hi_ats  = (hi_edge['_backed'] == (hi_edge['_cover'] > 0)).mean() if len(hi_edge) else float('nan')
    vhi_edge = ats_df[ats_df['_edge'] >= 5.0]
    vhi_ats  = (vhi_edge['_backed'] == (vhi_edge['_cover'] > 0)).mean() if len(vhi_edge) else float('nan')

    # O/U
    ou_mask = df['vegas_total'].notna()
    ou_df   = df[ou_mask].copy()
    ou_df['_pred_total'] = total[ou_mask].values
    ou_df = ou_df[ou_df['actual_total'] != ou_df['vegas_total']]
    ou_rate = ((ou_df['_pred_total'] > ou_df['vegas_total']) ==
               (ou_df['actual_total'] > ou_df['vegas_total'])).mean() if len(ou_df) else float('nan')

    return {
        'mae':      mae,
        'within35': within35,
        'within7':  within7,
        'correct':  correct,
        'ats':      ats_rate,
        'hi_ats':   hi_ats,     # edge >= 3
        'vhi_ats':  vhi_ats,    # edge >= 5
        'ou':       ou_rate,
        'n_games':  len(df),
        'n_ats':    int(ats_mask.sum()),
        'n_hi_ats': len(hi_edge),
    }

# ══════════════════════════════════════════════════════════
# GRID SEARCH OPTIMIZATION
# ══════════════════════════════════════════════════════════
def run_optimization(df):
    print('\n' + '═' * 60)
    print('GRID SEARCH — scoreScale × homeCourt')
    print(f'Testing {len(SCORE_SCALES)} × {len(HOME_COURTS)} = '
          f'{len(SCORE_SCALES)*len(HOME_COURTS)} combinations')
    print('═' * 60)

    results = []
    for ss in tqdm(SCORE_SCALES, desc='scoreScale'):
        for hc in HOME_COURTS:
            m = score_params(df, ss, hc)
            results.append({'score_scale': ss, 'home_court': hc, **m})

    res_df = pd.DataFrame(results)
    res_df.to_csv(DATA_DIR / 'optimization_results.csv', index=False)

    def pct(v):
        return f'{v*100:.1f}%' if not math.isnan(v) else '—'

    print('\n── Top 5 by lowest MAE ─────────────────────────────────')
    top_mae = res_df.nsmallest(5, 'mae')
    for _, r in top_mae.iterrows():
        print(f'  scoreScale={r.score_scale:.2f}  homeCourt={r.home_court:.1f}  '
              f'MAE={r.mae:.2f}  correct={pct(r.correct)}  '
              f'ATS={pct(r.ats)} ({r.n_ats:.0f}g)  hi_ATS={pct(r.hi_ats)}')

    min_games = 150
    good = res_df[res_df['n_ats'] >= min_games]
    if not good.empty:
        print(f'\n── Top 5 by ATS rate (min {min_games} games) ────────────────')
        for _, r in good.nlargest(5, 'ats').iterrows():
            print(f'  scoreScale={r.score_scale:.2f}  homeCourt={r.home_court:.1f}  '
                  f'ATS={pct(r.ats)}  hi_ATS={pct(r.hi_ats)}  vhi_ATS={pct(r.vhi_ats)}  '
                  f'MAE={r.mae:.2f}')

        print(f'\n── Top 5 by high-edge ATS (|model−vegas| ≥ 3 pts) ─────────')
        for _, r in good.nlargest(5, 'hi_ats').iterrows():
            print(f'  scoreScale={r.score_scale:.2f}  homeCourt={r.home_court:.1f}  '
                  f'ATS={pct(r.ats)}  hi_ATS={pct(r.hi_ats)} ({r.n_hi_ats:.0f}g)  '
                  f'vhi_ATS={pct(r.vhi_ats)}  MAE={r.mae:.2f}')

    best_mae = res_df.loc[res_df['mae'].idxmin()]
    best_ats = good.loc[good['ats'].idxmax()] if not good.empty else best_mae

    print('\n' + '═' * 60)
    print('RECOMMENDED PARAMETERS')
    print('  (Best MAE — most accurate raw predictions)')
    print(f'    scoreScale = {best_mae.score_scale:.2f}')
    print(f'    homeCourt  = {best_mae.home_court:.1f}')
    print()
    print('  (Best ATS — optimized for beating the line)')
    print(f'    scoreScale = {best_ats.score_scale:.2f}')
    print(f'    homeCourt  = {best_ats.home_court:.1f}')
    print()
    print('  Update cfg defaults in index.html:')
    print(f'    scoreScale: {best_ats.score_scale:.2f}')
    print(f'    homeCourt:  {best_ats.home_court:.1f}')
    print('═' * 60)
    print(f'\nFull grid results saved to {DATA_DIR / "optimization_results.csv"}')

    return res_df

# ══════════════════════════════════════════════════════════
# DIAGNOSTICS  (run after optimization to understand the data)
# ══════════════════════════════════════════════════════════
def run_diagnostics(df, score_scale=1.06, home_court=3.5):
    print('\n' + '═' * 60)
    print(f'DIAGNOSTICS  (scoreScale={score_scale}, homeCourt={home_court})')
    print('═' * 60)

    m = score_params(df, score_scale, home_court)

    print(f'Games:       {m["n_games"]:,}')
    print(f'MAE:         {m["mae"]:.2f} pts')
    print(f'Within 3.5:  {m["within35"]*100:.1f}%')
    print(f'Within 7:    {m["within7"]*100:.1f}%')
    print(f'Correct side:{m["correct"]*100:.1f}%')
    print(f'ATS:         {m["ats"]*100:.1f}%  ({m["n_ats"]} games)')
    print(f'ATS ≥3 edge: {m["hi_ats"]*100:.1f}%  ({m["n_hi_ats"]} games)')
    print(f'O/U:         {m["ou"]*100:.1f}%')
    print(f'\n52.4% = breakeven at -110 juice')

    # ATS by edge bucket
    poss   = np.sqrt(df['h_adjT'] * df['a_adjT'])
    h_sc   = (df['h_adjO'] * df['a_adjD'] / 100) * poss / 100 * score_scale
    a_sc   = (df['a_adjO'] * df['h_adjD'] / 100) * poss / 100 * score_scale
    hc_arr = np.where(df['neutral'].astype(bool), 0.0, home_court)
    margin = (h_sc - a_sc) + hc_arr

    ats_df = df[df['vegas_spread'].notna()].copy()
    ats_df['_margin'] = margin[ats_df.index].values
    ats_df['_cover']  = ats_df['actual_margin'] + ats_df['vegas_spread']
    ats_df['_backed'] = ats_df['_margin'] >= 0
    ats_df['_edge']   = np.abs(ats_df['_margin'] + ats_df['vegas_spread'])
    ats_df = ats_df[ats_df['_cover'] != 0]

    print('\nATS breakdown by edge size:')
    print(f'  {"Bucket":<14} {"Games":>6} {"W-L":>8} {"ATS%":>7} {"vs BE":>8}')
    for label, lo, hi in [
        ('≥ 5 pts',  5.0, 999),
        ('3–5 pts',  3.0, 5.0),
        ('1.5–3',    1.5, 3.0),
        ('< 1.5',    0.0, 1.5),
    ]:
        bk = ats_df[(ats_df['_edge'] >= lo) & (ats_df['_edge'] < hi)]
        if bk.empty:
            continue
        correct = (bk['_backed'] == (bk['_cover'] > 0))
        w = correct.sum()
        l = len(correct) - w
        pct = w / len(correct) * 100
        vs_be = pct - 52.4
        sign = '+' if vs_be >= 0 else ''
        print(f'  {label:<14} {len(bk):>6} {w:>4}-{l:<4} {pct:>6.1f}% {sign}{vs_be:>6.1f}%')

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('TrueLine LS — Model Optimization')
    print('=' * 60)

    if ODDS_API_KEY == 'YOUR_KEY_HERE':
        print('ERROR: Set ODDS_API_KEY at the top of this script.')
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_DIR / 'dataset.csv'
    if dataset_path.exists():
        print(f'Loading cached dataset from {dataset_path}...')
        df = pd.read_csv(dataset_path)
        print(f'Loaded {len(df):,} games '
              f'({df["vegas_spread"].notna().sum():,} with spreads)')
    else:
        print('No cached dataset found — fetching from APIs...')
        df = build_dataset()

    if df is None or df.empty:
        print('No data. Check your API key and internet connection.')
        sys.exit(1)

    run_diagnostics(df)
    run_optimization(df)
