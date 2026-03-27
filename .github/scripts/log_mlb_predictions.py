"""
log_mlb_predictions.py
Runs after MLB odds are fetched. For each game in today's odds Gist file:
  - fetches probable pitcher stats from MLB Stats API
  - runs the same model math as index.html
  - appends the game to mlb-log-YYYY-MM-DD.json (skips games already logged)
Output Gist file: mlb-log-YYYY-MM-DD.json
"""

import json, math, re, urllib.request, os, sys
from datetime import datetime, timezone

# ── Env ───────────────────────────────────────────────────────────────────────
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')

GIST_USER = 'loganthein'

# ── Static model data (mirrors index.html) ───────────────────────────────────
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

# ── Helpers (mirrors index.html) ──────────────────────────────────────────────
def norm(name):
    return re.sub(r'[^a-z]', '', name.lower()) if name else ''

def get_park_factor(home_team):
    return MLB_PARK_FACTORS.get(norm(home_team), 1.00)

def rate_offense(team, pitcher_hand):
    entry = MLB_WRC.get(norm(team))
    if not entry:
        return 0.0
    if pitcher_hand == 'L':
        wrc = entry['vsL']
    elif pitcher_hand == 'R':
        wrc = entry['vsR']
    else:
        wrc = (entry['vsL'] + entry['vsR']) / 2
    return round((wrc - 100) / 20 * 100) / 100

def parse_pitcher(p):
    if not p:
        return None
    full  = p.get('fullName', '')
    parts = full.strip().split()
    short = f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) >= 2 else full
    stats_list = p.get('stats', [])
    season_entry = next(
        (s for s in stats_list
         if s.get('group', {}).get('displayName') == 'pitching'
         and s.get('type', {}).get('displayName') == 'season'),
        None
    )
    season = (season_entry or {}).get('splits', [{}])[0].get('stat', {}) if season_entry else {}
    ip  = float(season.get('inningsPitched', 0) or 0)
    era = float(season['era'])  if season.get('era')  is not None else None
    fip = float(season['fip'])  if season.get('fip')  is not None else None
    whip = float(season['whip']) if season.get('whip') is not None else None
    k9  = round(float(season['strikeOuts']) / ip * 9, 1) if ip > 0 and season.get('strikeOuts') is not None else None
    bb9 = round(float(season['baseOnBalls']) / ip * 9, 1) if ip > 0 and season.get('baseOnBalls') is not None else None
    hand_obj = p.get('pitchHand') or {}
    hand = hand_obj.get('code') if isinstance(hand_obj, dict) else None
    return {
        'name':    short or 'TBD',
        'hand':    hand,
        'era':     round(era, 2)  if era  is not None else None,
        'fip':     round(fip, 2)  if fip  is not None else None,
        'fipRaw':  fip,
        'eraRaw':  era,
        'whip':    round(whip, 2) if whip is not None else None,
        'k9':      k9,
        'bb9':     bb9,
    }

def rate_pitcher(p):
    if not p:
        return None
    base = p.get('fipRaw') if p.get('fipRaw') is not None else p.get('eraRaw')
    if base is None:
        return None
    return round((4.50 - base) * 100) / 100

def prob_to_ml(prob):
    if prob is None:
        return None
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    return round(((1 - prob) / prob) * 100)

def ml_to_prob(ml):
    if ml is None:
        return None
    return 100 / (ml + 100) if ml > 0 else (-ml) / (-ml + 100)

def calc_total(home, away, hp, ap, pf):
    ho = rate_offense(home, ap.get('hand') if ap else None)
    ao = rate_offense(away, hp.get('hand') if hp else None)
    hp_r = rate_pitcher(hp) or 0
    ap_r = rate_pitcher(ap) or 0
    total = 9.0 - (hp_r + ap_r) + (ho + ao) * 2.0
    total *= pf
    return round(max(5.5, min(14.0, total)) * 10) / 10

def calc_win_prob(home, away, hp, ap, pf):
    ho = rate_offense(home, ap.get('hand') if ap else None)
    ao = rate_offense(away, hp.get('hand') if hp else None)
    hp_r = rate_pitcher(hp) or 0
    ap_r = rate_pitcher(ap) or 0
    home_runs = (9.0 / 2 - hp_r + ho * 2.0) * pf
    away_runs = (9.0 / 2 - ap_r + ao * 2.0) * pf * 0.95
    run_diff  = home_runs - away_runs
    home_wp   = 1 / (1 + math.exp(-run_diff * 0.35))
    home_wp  += 0.025
    home_wp   = max(0.30, min(0.75, home_wp))
    return round(home_wp * 100), round((1 - home_wp) * 100), run_diff

def calc_rl_cover(run_diff, home_favored):
    fav_diff = run_diff if home_favored else -run_diff
    return 1 / (1 + math.exp(-(fav_diff - 1.5) * 0.45))

def calc_f5(home, away, hp, ap, pf):
    ho = rate_offense(home, ap.get('hand') if ap else None)
    ao = rate_offense(away, hp.get('hand') if hp else None)
    hp_r = rate_pitcher(hp) or 0
    ap_r = rate_pitcher(ap) or 0
    scale = 5 / 9
    total = (9.0 - (hp_r + ap_r) + (ho + ao) * 2.0) * scale * pf
    total = round(max(2.5, min(7.5, total)) * 10) / 10
    home_runs = (9.0 / 2 - hp_r + ho * 2.0) * scale * pf
    away_runs = (9.0 / 2 - ap_r + ao * 2.0) * scale * pf * 0.97
    run_diff  = home_runs - away_runs
    home_wp   = 1 / (1 + math.exp(-run_diff * 0.45))
    home_wp  += 0.015
    home_wp   = max(0.30, min(0.72, home_wp))
    return {
        'total':       total,
        'homeWinProb': round(home_wp * 100),
        'awayWinProb': round((1 - home_wp) * 100),
        'homeModelML': prob_to_ml(home_wp),
        'awayModelML': prob_to_ml(1 - home_wp),
    }

# ── Gist helpers ──────────────────────────────────────────────────────────────
def gist_fetch(filename):
    url = f'https://gist.githubusercontent.com/{GIST_USER}/{gist_id}/raw/{filename}'
    try:
        req = urllib.request.Request(url, headers={'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'Note: could not fetch {filename}: {e}')
        return None

def gist_patch(filename, data):
    body = json.dumps({'files': {filename: {'content': json.dumps(data, indent=2)}}}).encode()
    req  = urllib.request.Request(
        f'https://api.github.com/gists/{gist_id}',
        data=body, method='PATCH',
        headers={
            'Authorization': f'token {gist_pat}',
            'Content-Type':  'application/json',
            'Accept':        'application/vnd.github.v3+json',
            'User-Agent':    'log-mlb-predictions',
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status

# ── Fetch pitcher data from MLB Stats API ─────────────────────────────────────
def fetch_pitchers(iso_date):
    """Returns dict: 'normAway|normHome' → {away: parsed, home: parsed}"""
    sched_url = (
        f'https://statsapi.mlb.com/api/v1/schedule'
        f'?sportId=1&date={iso_date}&gameType=R&hydrate=probablePitcher'
    )
    try:
        req = urllib.request.Request(sched_url, headers={'User-Agent': 'mlb-log-script'})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f'Warning: MLB schedule fetch failed: {e}')
        return {}

    stubs = []
    pitcher_ids = set()
    for date_entry in d.get('dates', []):
        for game in date_entry.get('games', []):
            away_p = game.get('teams', {}).get('away', {}).get('probablePitcher') or None
            home_p = game.get('teams', {}).get('home', {}).get('probablePitcher') or None
            if away_p and away_p.get('id'):
                pitcher_ids.add(away_p['id'])
            if home_p and home_p.get('id'):
                pitcher_ids.add(home_p['id'])
            stubs.append({
                'awayName': game.get('teams', {}).get('away', {}).get('team', {}).get('name', ''),
                'homeName': game.get('teams', {}).get('home', {}).get('team', {}).get('name', ''),
                'awayP': away_p,
                'homeP': home_p,
            })

    stats_by_id = {}
    if pitcher_ids:
        ids_param = ','.join(str(i) for i in pitcher_ids)
        people_url = (
            f'https://statsapi.mlb.com/api/v1/people?personIds={ids_param}'
            f'&hydrate=stats(group=pitching,type=season,season=2025)'
        )
        try:
            req2 = urllib.request.Request(people_url, headers={'User-Agent': 'mlb-log-script'})
            with urllib.request.urlopen(req2, timeout=10) as r2:
                sd = json.loads(r2.read())
            for person in sd.get('people', []):
                stats_by_id[person['id']] = {
                    'stats':    person.get('stats', []),
                    'pitchHand': person.get('pitchHand'),
                }
        except Exception as e:
            print(f'Warning: MLB people fetch failed: {e}')

    result = {}
    for stub in stubs:
        away_p = stub['awayP']
        home_p = stub['homeP']
        if away_p and away_p.get('id') and away_p['id'] in stats_by_id:
            away_p['stats']    = stats_by_id[away_p['id']]['stats']
            away_p['pitchHand'] = away_p.get('pitchHand') or stats_by_id[away_p['id']]['pitchHand']
        if home_p and home_p.get('id') and home_p['id'] in stats_by_id:
            home_p['stats']    = stats_by_id[home_p['id']]['stats']
            home_p['pitchHand'] = home_p.get('pitchHand') or stats_by_id[home_p['id']]['pitchHand']
        key = f"{norm(stub['awayName'])}|{norm(stub['homeName'])}"
        result[key] = {
            'away': parse_pitcher(away_p),
            'home': parse_pitcher(home_p),
        }
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
odds_data = gist_fetch(f'mlb-odds-{date_str}.json')
if not odds_data:
    print(f'No mlb-odds-{date_str}.json found in Gist — nothing to log.')
    sys.exit(0)

existing_log = gist_fetch(f'mlb-log-{date_str}.json') or []
logged_ids   = {entry['game_id'] for entry in existing_log if 'game_id' in entry}

pitchers = fetch_pitchers(date_str)
logged_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
added = 0

for game in odds_data:
    gid = game.get('id')
    if not gid or gid in logged_ids:
        continue  # already logged

    home = game.get('home_team', '')
    away = game.get('away_team', '')
    pitch_key = f'{norm(away)}|{norm(home)}'
    p = pitchers.get(pitch_key, {})
    hp = p.get('home')
    ap = p.get('away')
    pf = get_park_factor(home)

    # Model calculations
    total     = calc_total(home, away, hp, ap, pf)
    home_wp, away_wp, run_diff = calc_win_prob(home, away, hp, ap, pf)
    home_fav  = home_wp >= away_wp
    rl_cover  = calc_rl_cover(run_diff, home_fav)
    f5        = calc_f5(home, away, hp, ap, pf)

    # Odds extraction
    markets  = game.get('bookmakers', [{}])[0].get('markets', []) if game.get('bookmakers') else []
    def find_market(key):
        return next((m for m in markets if m['key'] == key), None)
    def outcome(mkt, name=None, point=None):
        if not mkt:
            return None
        for o in mkt.get('outcomes', []):
            if name and o.get('name') != name:
                continue
            if point is not None and o.get('point') != point:
                continue
            return o
        return None

    h2h      = find_market('h2h')
    spreads  = find_market('spreads')
    totals   = find_market('totals')
    f5h2h    = find_market('h2h_h1')
    f5totals = find_market('totals_h1')

    home_out  = outcome(h2h, name=home)
    away_out  = outcome(h2h, name=away)
    rl_home   = outcome(spreads, name=home)
    rl_away   = outcome(spreads, name=away)
    total_over = outcome(totals, name='Over')
    f5_home   = outcome(f5h2h, name=home)
    f5_away   = outcome(f5h2h, name=away)
    f5_over   = outcome(f5totals, name='Over')

    home_ml  = home_out['price']  if home_out  else None
    away_ml  = away_out['price']  if away_out  else None
    rl_home_point = rl_home['point'] if rl_home else None
    rl_home_ml    = rl_home['price'] if rl_home else None
    rl_away_point = rl_away['point'] if rl_away else None
    rl_away_ml    = rl_away['price'] if rl_away else None

    # Model's favored team run line ML (only valid if they have -1.5)
    fav_rl_point = rl_home_point if home_fav else rl_away_point
    fav_rl_ml    = rl_home_ml    if home_fav else rl_away_ml
    model_fav_rl_ml = fav_rl_ml if fav_rl_point == -1.5 else None

    # Edge calculations (vs. vig-adjusted implied prob)
    home_imp   = ml_to_prob(home_ml)
    rl_imp     = ml_to_prob(model_fav_rl_ml)
    f5home_imp = ml_to_prob(f5_home['price'] if f5_home else None)
    f5tot_line = f5_over['point'] if f5_over else None

    home_edge   = round((home_wp / 100 - home_imp) * 100)  if home_imp   is not None else None
    rl_edge     = round((rl_cover - rl_imp) * 100)         if rl_imp     is not None else None
    f5home_edge = round((f5['homeWinProb'] / 100 - f5home_imp) * 100) if f5home_imp is not None else None
    f5tot_edge  = round((f5['total'] - f5tot_line) * 10) / 10         if f5tot_line is not None else None

    entry = {
        'game_id':   gid,
        'game':      f'{away} @ {home}',
        'home':      home,
        'away':      away,
        'commence':  game.get('commence_time'),
        'logged_at': logged_at,
        'pitchers': {
            'home': hp,
            'away': ap,
        },
        'model': {
            'total':       total,
            'homeWinProb': home_wp,
            'awayWinProb': away_wp,
            'homeModelML': prob_to_ml(home_wp / 100),
            'awayModelML': prob_to_ml(away_wp / 100),
            'runDiff':     round(run_diff * 100) / 100,
            'rlFavTeam':   home if home_fav else away,
            'rlCoverPct':  round(rl_cover * 100),
            'f5Total':     f5['total'],
            'f5HomeWinProb': f5['homeWinProb'],
            'f5AwayWinProb': f5['awayWinProb'],
            'f5HomeModelML': f5['homeModelML'],
            'f5AwayModelML': f5['awayModelML'],
        },
        'vegas': {
            'homeML':      home_ml,
            'awayML':      away_ml,
            'rlHomePoint': rl_home_point,
            'rlHomeML':    rl_home_ml,
            'rlAwayPoint': rl_away_point,
            'rlAwayML':    rl_away_ml,
            'totalLine':   total_over['point'] if total_over else None,
            'totalOverML': total_over['price'] if total_over else None,
            'f5HomeML':    f5_home['price'] if f5_home else None,
            'f5AwayML':    f5_away['price'] if f5_away else None,
            'f5TotalLine': f5tot_line,
            'f5TotalOverML': f5_over['price'] if f5_over else None,
        },
        'edges': {
            'homeEdge':   home_edge,
            'rlEdge':     rl_edge,
            'f5HomeEdge': f5home_edge,
            'f5TotEdge':  f5tot_edge,
        },
        'result': None,  # filled in later: {homeScore, awayScore, winner, totalRuns, f5HomeRuns, f5AwayRuns}
    }

    existing_log.append(entry)
    logged_ids.add(gid)
    added += 1
    print(f'Logged: {away} @ {home}')

if added == 0:
    print('No new games to log.')
    sys.exit(0)

log_filename = f'mlb-log-{date_str}.json'
print(f'Writing {log_filename} ({len(existing_log)} total entries, {added} new)...')
try:
    status = gist_patch(log_filename, existing_log)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
