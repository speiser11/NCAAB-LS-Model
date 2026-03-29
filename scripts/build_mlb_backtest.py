"""
build_mlb_backtest.py

One-time script to build a historical MLB backtesting dataset.

Usage:
  ODDS_API_KEY=your_key python scripts/build_mlb_backtest.py

Output:
  mlb_backtest.csv  — one row per completed game with all model inputs + predictions + actuals

Data sources:
  - The Odds API /v4/historical endpoint (pre-game lines snapshot)
  - MLB Stats API (schedules, scores, probable pitchers, pitcher stats, bullpen stats)
  - Static: wRC+ splits and park factors (same tables as live model)

Credit estimate:
  - ~3 credits/date × ~550 dates (2023–2025) = ~1,650 credits total
  - F5 per-event disabled — too expensive at scale

Notes:
  - Uses full-season pitcher/bullpen stats (not in-season rolling). Fine for testing
    whether the variables are predictive; not a perfect real-time simulation.
  - wRC+ splits are 2025 data used as proxy for 2024. Year-specific splits would
    be more rigorous but require manual FanGraphs pulls.
  - Historical odds snapshot is taken at 13:00 UTC (9am ET) each date — morning
    lines, before most games start.
  - F5 results computed from per-inning linescores via MLB Stats API (free).
  - All intermediate data is cached to .mlb_backtest_cache/ so the script can
    be stopped and resumed without re-fetching.
"""

import json, os, sys, csv, time, math, re, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
CACHE_DIR    = Path('.mlb_backtest_cache')
OUTPUT_FILE  = 'mlb_backtest.csv'

# 2024 regular season through end of 2025 regular season
# Opening day 2024 was March 20 (Seoul), domestic March 28
START_DATE = '2023-03-30'
END_DATE   = '2025-09-28'   # adjust when 2025 season ends

# Odds snapshot time: 13:00 UTC = 9am ET (safely pre-game morning lines)
SNAPSHOT_HOUR = 'T13:00:00Z'

# ── Static tables (mirrors index.html exactly) ────────────────────────────────
MLB_PARK_FACTORS = {
    'coloradorockies':      1.18,
    'cincinnatireds':       1.08,
    'newyorkyankees':       1.07,
    'bostonredsox':         1.06,
    'baltimoreorioles':     1.05,
    'texasrangers':         1.05,
    'chicagocubs':          1.04,
    'philadelphiaphillies': 1.04,
    'atlantabraves':        1.03,
    'arizonadiamondbacks':  1.02,
    'kansascityroyals':     1.02,
    'clevelandguardians':   1.01,
    'milwaukeebrewers':     1.01,
    'detroittigers':        1.01,
    'chicagowhitesox':      1.01,
    'minnesotatwins':       1.00,
    'torontobluejays':      1.00,
    'houstonastros':        1.00,
    'losangelesangels':     1.00,
    'stlouiscardinals':     0.99,
    'newyorkmets':          0.98,
    'washingtonnationals':  0.98,
    'losangelesdodgers':    0.97,
    'pittsburghpirates':    0.97,
    'miamimarlins':         0.97,
    'oaklandathletics':     0.97,
    'athletics':            0.97,
    'seattlemariners':      0.96,
    'tampabayrays':         0.94,
    'sandiegopadres':       0.94,
    'sanfranciscogiants':   0.93,
}

# wRC+ vs LHP / RHP — 2025 FanGraphs splits (proxy for both seasons)
MLB_WRC = {
    'losangelesdodgers':    {'vsL': 111, 'vsR': 114},
    'newyorkyankees':       {'vsL': 121, 'vsR': 118},
    'philadelphiaphillies': {'vsL': 106, 'vsR': 110},
    'atlantabraves':        {'vsL':  99, 'vsR': 101},
    'baltimoreorioles':     {'vsL':  87, 'vsR': 100},
    'minnesotatwins':       {'vsL':  97, 'vsR':  97},
    'bostonredsox':         {'vsL': 103, 'vsR': 103},
    'clevelandguardians':   {'vsL':  81, 'vsR':  90},
    'houstonastros':        {'vsL': 107, 'vsR':  98},
    'arizonadiamondbacks':  {'vsL': 105, 'vsR': 110},
    'sandiegopadres':       {'vsL':  96, 'vsR': 105},
    'milwaukeebrewers':     {'vsL': 107, 'vsR': 107},
    'torontobluejays':      {'vsL': 111, 'vsR': 113},
    'newyorkmets':          {'vsL':  98, 'vsR': 118},
    'chicagocubs':          {'vsL': 110, 'vsR': 110},
    'texasrangers':         {'vsL':  83, 'vsR':  95},
    'tampabayrays':         {'vsL':  85, 'vsR': 104},
    'cincinnatireds':       {'vsL':  79, 'vsR':  97},
    'kansascityroyals':     {'vsL':  80, 'vsR':  97},
    'seattlemariners':      {'vsL': 108, 'vsR': 114},
    'detroittigers':        {'vsL': 114, 'vsR':  98},
    'sanfranciscogiants':   {'vsL':  78, 'vsR': 104},
    'stlouiscardinals':     {'vsL':  97, 'vsR':  96},
    'coloradorockies':      {'vsL':  75, 'vsR':  74},
    'washingtonnationals':  {'vsL':  78, 'vsR':  99},
    'pittsburghpirates':    {'vsL':  76, 'vsR':  84},
    'losangelesangels':     {'vsL':  91, 'vsR':  92},
    'oaklandathletics':     {'vsL': 102, 'vsR': 106},
    'athletics':            {'vsL': 102, 'vsR': 106},
    'miamimarlins':         {'vsL':  82, 'vsR': 102},
    'chicagowhitesox':      {'vsL':  93, 'vsR':  87},
}

# ── Normalization ─────────────────────────────────────────────────────────────
def norm(name):
    """Same as normMLBTeam() in index.html"""
    return re.sub(r'[^a-z]', '', name.lower()) if name else ''

# ── Model math (exact mirror of index.html) ───────────────────────────────────
def rate_pitcher(p):
    if not p:
        return 0
    base = p.get('fipRaw') or p.get('eraRaw')
    if base is None:
        return 0
    return round(4.50 - base, 2)

def rate_offense(team_name, pitcher_hand):
    key = norm(team_name)
    row = MLB_WRC.get(key)
    if not row:
        return 0
    if pitcher_hand == 'L':
        wrc = row['vsL']
    elif pitcher_hand == 'R':
        wrc = row['vsR']
    else:
        wrc = (row['vsL'] + row['vsR']) / 2
    return round((wrc - 100) / 20, 2)

def get_park_factor(home_team):
    return MLB_PARK_FACTORS.get(norm(home_team), 1.00)

def calc_total(hp, ap, ho, ao, home_bp, away_bp, pf):
    home_eff = hp * (5/9) + home_bp * (4/9)
    away_eff = ap * (5/9) + away_bp * (4/9)
    total = 9.0 - home_eff - away_eff + (ho + ao) * 2.0
    total *= pf
    return max(5.5, min(14.0, round(total, 1)))

def calc_win_prob(hp, ap, ho, ao, home_bp, away_bp, pf):
    home_eff = hp * (5/9) + home_bp * (4/9)
    away_eff = ap * (5/9) + away_bp * (4/9)
    home_runs = (9.0 / 2) - away_eff + ho * 2.0
    away_runs = (9.0 / 2) - home_eff + ao * 2.0
    home_runs *= pf
    away_runs *= pf * 0.95
    run_diff  = home_runs - away_runs
    raw_prob  = 1 / (1 + math.exp(-run_diff * 0.18))
    home_prob = min(0.75, max(0.30, raw_prob + 0.025))
    return home_prob, run_diff

def calc_rl_cover(run_diff, home_favored):
    fav_diff = run_diff if home_favored else -run_diff
    return 1 / (1 + math.exp(-(fav_diff - 1.5) * 0.45))

def prob_to_ml(p):
    p = max(0.01, min(0.99, p))
    if p >= 0.5:
        return round(-p / (1 - p) * 100)
    return round((1 - p) / p * 100)

def ml_to_prob(ml):
    if ml is None:
        return None
    ml = float(ml)
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)

F5_SCALE = 5 / 9

def calc_f5(hp, ap, ho, ao, pf):
    """F5 model: starter quality only, no bullpen, scaled to 5 innings.
    Mirrors calcMLBF5() in index.html exactly."""
    total = (9.0 - (hp + ap) + (ho + ao) * 2.0) * F5_SCALE * pf
    total = max(2.5, min(7.5, round(total, 1)))
    home_runs = ((9.0 / 2) - ap + ho * 2.0) * F5_SCALE * pf
    away_runs = ((9.0 / 2) - hp + ao * 2.0) * F5_SCALE * pf * 0.97
    run_diff  = home_runs - away_runs
    home_win  = min(0.72, max(0.30, 1 / (1 + math.exp(-run_diff * 0.45)) + 0.015))
    return {
        'total':         total,
        'home_win_prob': round(home_win * 100, 1),
        'run_diff':      round(run_diff, 2),
        'home_ml':       prob_to_ml(home_win),
        'away_ml':       prob_to_ml(1 - home_win),
    }

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def http_get(url, params=None):
    if params:
        url = url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'mlb-backtest'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def odds_get(path, params):
    params['apiKey'] = ODDS_API_KEY
    url = 'https://api.the-odds-api.com' + path + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'mlb-backtest'})
    with urllib.request.urlopen(req, timeout=20) as r:
        remaining = r.headers.get('x-requests-remaining', '?')
        return json.loads(r.read()), remaining

# ── Cache helpers ─────────────────────────────────────────────────────────────
def cache_read(key):
    p = CACHE_DIR / f'{key}.json'
    if p.exists():
        return json.loads(p.read_text())
    return None

def cache_write(key, data):
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f'{key}.json').write_text(json.dumps(data))

F5_MARKET_KEYS = ('h2h_1st_5_innings', 'spreads_1st_5_innings', 'totals_1st_5_innings')

# ── Fetch: historical odds ────────────────────────────────────────────────────
def fetch_historical_odds(date_str):
    """Fetch pre-game odds snapshot. Tries all 6 markets including F5 in bulk.
    If the bulk endpoint rejects F5 keys (422), falls back to base markets only
    and then fetches F5 per-event using the historical events endpoint."""
    cached = cache_read(f'odds_{date_str}')
    if cached is not None:
        return cached

    snapshot_ts  = f'{date_str}{SNAPSHOT_HOUR}'
    all_markets  = 'h2h,spreads,totals,h2h_1st_5_innings,spreads_1st_5_innings,totals_1st_5_innings'
    base_markets = 'h2h,spreads,totals'

    # Try bulk with F5 first
    f5_in_bulk = False
    try:
        result, remaining = odds_get('/v4/historical/sports/baseball_mlb/odds', {
            'date':       snapshot_ts,
            'regions':    'us',
            'markets':    all_markets,
            'oddsFormat': 'american',
        })
        games = result.get('data', [])
        f5_in_bulk = True
        print(f'  Odds (with F5 bulk): {len(games)} games — credits remaining: {remaining}')
    except Exception as e:
        print(f'  Bulk F5 failed ({e}), falling back to base markets...')
        try:
            result, remaining = odds_get('/v4/historical/sports/baseball_mlb/odds', {
                'date':       snapshot_ts,
                'regions':    'us',
                'markets':    base_markets,
                'oddsFormat': 'american',
            })
            games = result.get('data', [])
            print(f'  Odds (base only): {len(games)} games — credits remaining: {remaining}')
        except Exception as e2:
            print(f'  Odds fetch failed for {date_str}: {e2}')
            return []

    time.sleep(0.5)

    # F5 per-event disabled — too expensive (345 credits/day across a full season)
    # F5 model predictions + actual F5 results are still captured; just no Vegas F5 lines.

    cache_write(f'odds_{date_str}', games)
    return games

# ── Fetch: MLB schedule + scores ─────────────────────────────────────────────
def fetch_schedule(date_str):
    cached = cache_read(f'schedule_{date_str}')
    if cached is not None:
        return cached

    try:
        data = http_get(
            'https://statsapi.mlb.com/api/v1/schedule',
            {
                'sportId':  1,
                'date':     date_str,
                'gameType': 'R',
                'hydrate':  'probablePitcher,linescore',
            }
        )
        games = []
        for date_obj in data.get('dates', []):
            for g in date_obj.get('games', []):
                games.append(g)
        cache_write(f'schedule_{date_str}', games)
        return games
    except Exception as e:
        print(f'  Schedule fetch failed for {date_str}: {e}')
        return []

# ── Fetch: pitcher season stats ───────────────────────────────────────────────
_pitcher_mem = {}

def fetch_pitcher_stats(person_id, season):
    key = f'p_{person_id}_{season}'
    if key in _pitcher_mem:
        return _pitcher_mem[key]
    cached = cache_read(key)
    if cached is not None:
        _pitcher_mem[key] = cached
        return cached

    try:
        data = http_get(
            'https://statsapi.mlb.com/api/v1/people',
            {
                'personIds': person_id,
                'hydrate':   f'stats(group=pitching,type=season,season={season})',
            }
        )
        people = data.get('people', [])
        if not people:
            _pitcher_mem[key] = None
            return None
        p = people[0]
        # Find season stats split
        stat = {}
        for s in p.get('stats', []):
            if s.get('group', {}).get('displayName') == 'pitching':
                splits = s.get('splits', [])
                if splits:
                    stat = splits[0].get('stat', {})
                    break

        def safe_float(v, fallback=None):
            try:
                f = float(v)
                return f if f > 0 else fallback
            except (TypeError, ValueError):
                return fallback

        ip_str = stat.get('inningsPitched', '0') or '0'
        try:
            parts = str(ip_str).split('.')
            ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
        except Exception:
            ip = 0

        era_raw = safe_float(stat.get('era'))
        fip_raw = safe_float(stat.get('fip'))
        k9  = round(safe_float(stat.get('strikeOuts'), 0) / max(ip, 1) * 9, 1) if ip > 0 else None
        bb9 = round(safe_float(stat.get('baseOnBalls'), 0) / max(ip, 1) * 9, 1) if ip > 0 else None

        result = {
            'name':   p.get('fullName', ''),
            'hand':   p.get('pitchHand', {}).get('code') or None,
            'eraRaw': era_raw,
            'fipRaw': fip_raw,
            'era':    round(era_raw, 2) if era_raw else None,
            'fip':    round(fip_raw, 2) if fip_raw else None,
            'whip':   safe_float(stat.get('whip')),
            'k9':     k9,
            'bb9':    bb9,
            'ip':     round(ip, 1),
        }
        _pitcher_mem[key] = result
        cache_write(key, result)
        time.sleep(0.25)
        return result
    except Exception as e:
        print(f'    Pitcher {person_id}/{season} failed: {e}')
        _pitcher_mem[key] = None
        return None

# ── Fetch: bullpen season stats ───────────────────────────────────────────────
_bullpen_mem = {}

def fetch_bullpen_stats(team_id, season):
    key = f'bp_{team_id}_{season}'
    if key in _bullpen_mem:
        return _bullpen_mem[key]
    cached = cache_read(key)
    if cached is not None:
        _bullpen_mem[key] = cached
        return cached

    try:
        data = http_get(
            'https://statsapi.mlb.com/api/v1/stats',
            {
                'group':      'pitching',
                'sportId':    1,
                'season':     season,
                'teamId':     team_id,
                'playerPool': 'All',
                'limit':      80,
                'stats':      'season',
            }
        )
        splits = data.get('stats', [{}])[0].get('splits', [])
        total_ip  = 0
        era_sum   = 0
        for s in splits:
            st = s.get('stat', {})
            gp = int(st.get('gamesPlayed', 0) or 0)
            gs = int(st.get('gamesStarted', 0) or 0)
            if gp == 0:
                continue
            if gs > 0 and gs / gp >= 0.30:
                continue  # exclude starters
            ip_str = st.get('inningsPitched', '0') or '0'
            try:
                parts = str(ip_str).split('.')
                ip = int(parts[0]) + (int(parts[1]) / 3 if len(parts) > 1 and parts[1] else 0)
            except Exception:
                ip = 0
            if ip < 1:
                continue
            try:
                era = float(st.get('era') or 4.20)
            except Exception:
                era = 4.20
            era_sum  += era * ip
            total_ip += ip

        blended_era = round(era_sum / total_ip, 2) if total_ip >= 5 else 4.20
        result = {'score': round(4.20 - blended_era, 2), 'era': blended_era}
        _bullpen_mem[key] = result
        cache_write(key, result)
        time.sleep(0.2)
        return result
    except Exception as e:
        print(f'    Bullpen {team_id}/{season} failed: {e}')
        result = {'score': 0, 'era': 4.20}
        _bullpen_mem[key] = result
        return result

# ── Parse odds for a game ─────────────────────────────────────────────────────
def parse_f5_runs(game):
    """Extract first-5-inning run totals from MLB Stats API linescore."""
    innings = game.get('linescore', {}).get('innings', [])
    if len(innings) < 5:
        return None, None
    home_f5 = sum(float(inn.get('home', {}).get('runs', 0) or 0) for inn in innings[:5])
    away_f5 = sum(float(inn.get('away', {}).get('runs', 0) or 0) for inn in innings[:5])
    return home_f5, away_f5

def parse_odds(odds_games, home_name, away_name):
    hn = norm(home_name)
    an = norm(away_name)

    for og in odds_games:
        if not og.get('bookmakers'):
            continue
        oh = norm(og.get('home_team', ''))
        oa = norm(og.get('away_team', ''))
        if oh != hn or oa != an:
            continue

        markets = og['bookmakers'][0].get('markets', [])
        h2h      = next((m for m in markets if m['key'] == 'h2h'), None)
        spreads  = next((m for m in markets if m['key'] == 'spreads'), None)
        totals   = next((m for m in markets if m['key'] == 'totals'), None)
        f5h2h    = next((m for m in markets if m['key'] == 'h2h_1st_5_innings'), None)
        f5spreads= next((m for m in markets if m['key'] == 'spreads_1st_5_innings'), None)
        f5totals = next((m for m in markets if m['key'] == 'totals_1st_5_innings'), None)

        def outcome_price(market, name_norm):
            if not market:
                return None
            return next((o['price'] for o in market.get('outcomes', []) if norm(o['name']) == name_norm), None)

        def outcome_by_point(market, point):
            if not market:
                return None
            return next((o for o in market.get('outcomes', []) if o.get('point') == point), None)

        def outcome_by_name(market, name):
            if not market:
                return None
            return next((o for o in market.get('outcomes', []) if o.get('name') == name), None)

        rl_fav_out  = outcome_by_point(spreads, -1.5)
        total_over  = outcome_by_name(totals, 'Over')
        f5_rl_fav   = outcome_by_point(f5spreads, -0.5)
        f5_over     = outcome_by_name(f5totals, 'Over')

        return {
            'home_ml':        outcome_price(h2h, hn),
            'away_ml':        outcome_price(h2h, an),
            'rl_fav_team':    rl_fav_out['name'] if rl_fav_out else None,
            'rl_fav_ml':      rl_fav_out['price'] if rl_fav_out else None,
            'total_line':     total_over['point'] if total_over else None,
            'total_over_ml':  total_over['price'] if total_over else None,
            # F5
            'f5_home_ml':     outcome_price(f5h2h, hn),
            'f5_away_ml':     outcome_price(f5h2h, an),
            'f5_rl_fav_team': f5_rl_fav['name'] if f5_rl_fav else None,
            'f5_rl_fav_ml':   f5_rl_fav['price'] if f5_rl_fav else None,
            'f5_total_line':  f5_over['point'] if f5_over else None,
            'f5_total_over_ml': f5_over['price'] if f5_over else None,
        }
    return None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not ODDS_API_KEY:
        print('ERROR: set ODDS_API_KEY environment variable', file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(exist_ok=True)
    rows = []
    games_found = 0
    games_with_odds = 0

    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end   = datetime.strptime(END_DATE, '%Y-%m-%d')
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    if end > today:
        end = today - timedelta(days=1)  # only process completed dates

    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        season   = str(current.year)
        print(f'\n── {date_str} ──')

        schedule = fetch_schedule(date_str)
        if not schedule:
            current += timedelta(days=1)
            continue

        # Only process dates with final games
        final_games = [g for g in schedule
                       if g.get('status', {}).get('abstractGameState') == 'Final']
        if not final_games:
            current += timedelta(days=1)
            continue

        odds_data = fetch_historical_odds(date_str)

        for g in final_games:
            teams     = g.get('teams', {})
            home_info = teams.get('home', {})
            away_info = teams.get('away', {})
            home_name = home_info.get('team', {}).get('name', '')
            away_name = away_info.get('team', {}).get('name', '')
            home_id   = home_info.get('team', {}).get('id')
            away_id   = away_info.get('team', {}).get('id')
            home_score = int(home_info.get('score', 0) or 0)
            away_score = int(away_info.get('score', 0) or 0)

            if not home_name or not away_name:
                continue

            games_found += 1

            # Probable pitchers
            home_pid = home_info.get('probablePitcher', {}).get('id') if home_info.get('probablePitcher') else None
            away_pid = away_info.get('probablePitcher', {}).get('id') if away_info.get('probablePitcher') else None

            home_p = fetch_pitcher_stats(home_pid, season) if home_pid else None
            away_p = fetch_pitcher_stats(away_pid, season) if away_pid else None

            # Bullpen
            home_bp = fetch_bullpen_stats(home_id, season) if home_id else {'score': 0, 'era': 4.20}
            away_bp = fetch_bullpen_stats(away_id, season) if away_id else {'score': 0, 'era': 4.20}

            # Model inputs
            hp = rate_pitcher(home_p)
            ap = rate_pitcher(away_p)
            home_hand = home_p.get('hand') if home_p else None
            away_hand = away_p.get('hand') if away_p else None
            ho = rate_offense(home_name, away_hand)
            ao = rate_offense(away_name, home_hand)
            pf = get_park_factor(home_name)
            h_bp_score = home_bp.get('score', 0)
            a_bp_score = away_bp.get('score', 0)

            # Full-game model outputs
            model_total             = calc_total(hp, ap, ho, ao, h_bp_score, a_bp_score, pf)
            home_win_prob, run_diff = calc_win_prob(hp, ap, ho, ao, h_bp_score, a_bp_score, pf)
            model_home_fav          = home_win_prob >= 0.5
            rl_cover_prob           = calc_rl_cover(run_diff, model_home_fav)
            model_home_ml           = prob_to_ml(home_win_prob)
            model_away_ml           = prob_to_ml(1 - home_win_prob)

            # F5 model outputs (starter quality only, no bullpen)
            f5 = calc_f5(hp, ap, ho, ao, pf)

            # Vegas odds
            odds     = parse_odds(odds_data, home_name, away_name)
            has_odds = odds is not None
            if has_odds:
                games_with_odds += 1

            # ── Full-game results ─────────────────────────────────────────────
            actual_total  = home_score + away_score
            actual_winner = 'home' if home_score > away_score else 'away'
            ml_correct    = actual_winner == ('home' if model_home_fav else 'away')

            vegas_total       = odds.get('total_line') if odds else None
            total_ou          = None
            model_beats_total = None
            if vegas_total is not None:
                total_ou          = 'over' if actual_total > vegas_total else ('under' if actual_total < vegas_total else 'push')
                model_beats_total = 'over' if model_total > vegas_total else ('under' if model_total < vegas_total else 'push')

            if model_home_fav:
                model_rl_result = 'cover' if (home_score - away_score) >= 2 else 'no_cover'
            else:
                model_rl_result = 'cover' if (away_score - home_score) >= 2 else 'no_cover'

            rl_fav_team     = odds.get('rl_fav_team') if odds else None
            vegas_rl_result = None
            if rl_fav_team:
                fav_is_home = norm(rl_fav_team) == norm(home_name)
                if fav_is_home:
                    vegas_rl_result = 'cover' if (home_score - away_score) >= 2 else 'no_cover'
                else:
                    vegas_rl_result = 'cover' if (away_score - home_score) >= 2 else 'no_cover'

            # ── F5 results ────────────────────────────────────────────────────
            f5_home_actual, f5_away_actual = parse_f5_runs(g)
            f5_total_actual = round(f5_home_actual + f5_away_actual, 1) if f5_home_actual is not None else None
            f5_actual_winner = None
            if f5_home_actual is not None:
                if f5_home_actual > f5_away_actual:
                    f5_actual_winner = 'home'
                elif f5_away_actual > f5_home_actual:
                    f5_actual_winner = 'away'
                else:
                    f5_actual_winner = 'tie'

            f5_model_home_fav = f5['home_win_prob'] >= 50
            f5_ml_correct = None
            if f5_actual_winner and f5_actual_winner != 'tie':
                f5_ml_correct = f5_actual_winner == ('home' if f5_model_home_fav else 'away')

            f5_vegas_total   = odds.get('f5_total_line') if odds else None
            f5_total_ou      = None
            f5_model_dir     = None
            if f5_vegas_total is not None and f5_total_actual is not None:
                f5_total_ou  = 'over' if f5_total_actual > f5_vegas_total else ('under' if f5_total_actual < f5_vegas_total else 'push')
                f5_model_dir = 'over' if f5['total'] > f5_vegas_total else ('under' if f5['total'] < f5_vegas_total else 'push')

            f5_total_diff = round(f5['total'] - f5_vegas_total, 1) if f5_vegas_total is not None else None
            f5_tier       = 'take' if f5_total_diff and abs(f5_total_diff) >= 1.5 else ('lean' if f5_total_diff and abs(f5_total_diff) >= 1.0 else None)

            # ── Model edge flags (full-game) ──────────────────────────────────
            vegas_home_ml = odds.get('home_ml') if odds else None
            vegas_away_ml = odds.get('away_ml') if odds else None
            ml_imp_diff   = None
            if vegas_home_ml and vegas_away_ml:
                vfav_is_home = vegas_home_ml <= vegas_away_ml
                vfav_ml  = vegas_home_ml if vfav_is_home else vegas_away_ml
                mfav_ml  = model_home_ml if vfav_is_home else model_away_ml
                v_prob   = ml_to_prob(vfav_ml)
                m_prob   = ml_to_prob(mfav_ml)
                if v_prob and m_prob:
                    ml_imp_diff = round(abs(v_prob - m_prob) * 100, 1)

            total_diff = round(model_total - vegas_total, 1) if vegas_total else None
            rl_margin  = abs(run_diff)
            ml_tier    = 'take' if ml_imp_diff and ml_imp_diff >= 10 else ('lean' if ml_imp_diff and ml_imp_diff >= 7 else None)
            total_tier = 'take' if total_diff and abs(total_diff) >= 2.0 else ('lean' if total_diff and abs(total_diff) >= 1.5 else None)
            rl_tier    = 'take' if rl_margin > 2.5 else None

            rows.append({
                # Identity
                'date':            date_str,
                'season':          season,
                'home_team':       home_name,
                'away_team':       away_name,
                # Pitcher inputs
                'home_pitcher':    home_p['name'] if home_p else '',
                'away_pitcher':    away_p['name'] if away_p else '',
                'home_hand':       home_hand or '',
                'away_hand':       away_hand or '',
                'home_fip':        home_p.get('fip', '') if home_p else '',
                'away_fip':        away_p.get('fip', '') if away_p else '',
                'home_era':        home_p.get('era', '') if home_p else '',
                'away_era':        away_p.get('era', '') if away_p else '',
                'home_ip':         home_p.get('ip', '') if home_p else '',
                'away_ip':         away_p.get('ip', '') if away_p else '',
                'home_whip':       home_p.get('whip', '') if home_p else '',
                'away_whip':       away_p.get('whip', '') if away_p else '',
                'home_k9':         home_p.get('k9', '') if home_p else '',
                'away_k9':         away_p.get('k9', '') if away_p else '',
                # Offense / park / bullpen inputs
                'home_offense_score':   ho,
                'away_offense_score':   ao,
                'park_factor':          pf,
                'home_pitcher_quality': hp,
                'away_pitcher_quality': ap,
                'home_bullpen_score':   h_bp_score,
                'away_bullpen_score':   a_bp_score,
                'home_bullpen_era':     home_bp.get('era', ''),
                'away_bullpen_era':     away_bp.get('era', ''),
                # Full-game model outputs
                'model_total':          model_total,
                'model_home_win_pct':   round(home_win_prob * 100, 1),
                'model_away_win_pct':   round((1 - home_win_prob) * 100, 1),
                'model_run_diff':       round(run_diff, 2),
                'model_home_ml':        model_home_ml,
                'model_away_ml':        model_away_ml,
                'model_rl_cover_pct':   round(rl_cover_prob * 100, 1),
                # F5 model outputs
                'f5_model_total':       f5['total'],
                'f5_model_home_win_pct':f5['home_win_prob'],
                'f5_model_run_diff':    f5['run_diff'],
                'f5_model_home_ml':     f5['home_ml'],
                'f5_model_away_ml':     f5['away_ml'],
                # Full-game edge flags
                'ml_tier':              ml_tier or '',
                'total_tier':           total_tier or '',
                'rl_tier':              rl_tier or '',
                'f5_tier':              f5_tier or '',
                'ml_imp_diff_pct':      ml_imp_diff or '',
                'total_diff':           total_diff or '',
                'rl_margin':            round(rl_margin, 2),
                'f5_total_diff':        f5_total_diff or '',
                # Vegas lines
                'has_odds':             has_odds,
                'vegas_home_ml':        vegas_home_ml or '',
                'vegas_away_ml':        vegas_away_ml or '',
                'vegas_total':          vegas_total or '',
                'vegas_rl_fav':         rl_fav_team or '',
                'vegas_rl_fav_ml':      (odds.get('rl_fav_ml') if odds else '') or '',
                # F5 Vegas lines
                'f5_vegas_home_ml':     (odds.get('f5_home_ml') if odds else '') or '',
                'f5_vegas_away_ml':     (odds.get('f5_away_ml') if odds else '') or '',
                'f5_vegas_total':       f5_vegas_total or '',
                'f5_vegas_rl_fav':      (odds.get('f5_rl_fav_team') if odds else '') or '',
                'f5_vegas_rl_fav_ml':   (odds.get('f5_rl_fav_ml') if odds else '') or '',
                # Full-game results
                'home_score':           home_score,
                'away_score':           away_score,
                'actual_total':         actual_total,
                'actual_winner':        actual_winner,
                'ml_correct':           ml_correct,
                'total_ou_result':      total_ou or '',
                'model_total_dir':      model_beats_total or '',
                'model_rl_result':      model_rl_result,
                'vegas_rl_result':      vegas_rl_result or '',
                # F5 results
                'f5_home_runs':         f5_home_actual if f5_home_actual is not None else '',
                'f5_away_runs':         f5_away_actual if f5_away_actual is not None else '',
                'f5_total_runs':        f5_total_actual if f5_total_actual is not None else '',
                'f5_actual_winner':     f5_actual_winner or '',
                'f5_ml_correct':        f5_ml_correct if f5_ml_correct is not None else '',
                'f5_total_ou_result':   f5_total_ou or '',
                'f5_model_total_dir':   f5_model_dir or '',
            })

            has_f5_odds = bool(odds and odds.get('f5_total_line'))
            print(f'  {away_name} @ {home_name}: {away_score}-{home_score} '
                  f'(F5: {f5_away_actual}-{f5_home_actual}) '
                  f'model={round(home_win_prob*100)}%/{f5["total"]}r '
                  f'{"✓" if ml_correct else "✗"}ML '
                  f'{"[odds+F5]" if has_f5_odds else "[odds]" if has_odds else "[no odds]"}')

        current += timedelta(days=1)

    print(f'\n── Summary ──────────────────────────────')
    print(f'Games found:      {games_found}')
    print(f'Games with odds:  {games_with_odds}')
    print(f'Total rows:       {len(rows)}')

    if not rows:
        print('No rows to write.')
        return

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
