"""
fill_mlb_results.py
Runs each cycle after games start. For each entry in mlb-log-YYYY-MM-DD.json
that has result=null, checks ESPN for a final score and fills it in.
Also computes F5 runs from per-inning linescores.
"""

import json, re, urllib.request, os, sys
from datetime import datetime, timezone

# ── Env ───────────────────────────────────────────────────────────────────────
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')

GIST_USER  = 'loganthein'
ESPN_URL   = 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard'

def norm(name):
    return re.sub(r'[^a-z]', '', name.lower()) if name else ''

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
            'User-Agent':    'fill-mlb-results',
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status

# ── Fetch ESPN final scores ───────────────────────────────────────────────────
def fetch_espn_results(date_str):
    """
    Returns dict: norm(away)|norm(home) → {
        homeScore, awayScore, homeName, awayName,
        f5Home, f5Away,   # sum of first 5 inning linescores (None if not available)
        status            # 'final' | 'in_progress' | 'scheduled'
    }
    """
    date_param = date_str.replace('-', '')
    url = f'{ESPN_URL}?seasontype=2&dates={date_param}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'fill-mlb-results'})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f'ESPN fetch failed: {e}', file=sys.stderr)
        return {}

    results = {}
    for ev in d.get('events', []):
        comp = ev.get('competitions', [{}])[0]
        status_name = comp.get('status', {}).get('type', {}).get('name', '')

        if status_name == 'STATUS_FINAL':
            status = 'final'
        elif status_name == 'STATUS_IN_PROGRESS':
            status = 'in_progress'
        else:
            status = 'scheduled'

        cs   = comp.get('competitors', [])
        home = next((c for c in cs if c.get('homeAway') == 'home'), None)
        away = next((c for c in cs if c.get('homeAway') == 'away'), None)
        if not home or not away:
            continue

        home_name  = home.get('team', {}).get('displayName', '')
        away_name  = away.get('team', {}).get('displayName', '')
        home_score = int(home.get('score', 0) or 0)
        away_score = int(away.get('score', 0) or 0)

        # F5: sum first 5 linescores for each team
        def f5_runs(competitor):
            ls = competitor.get('linescores', [])
            if not ls:
                return None
            return sum(float(inning.get('value', 0) or 0) for inning in ls[:5])

        key = f'{norm(away_name)}|{norm(home_name)}'
        results[key] = {
            'homeName':  home_name,
            'awayName':  away_name,
            'homeScore': home_score,
            'awayScore': away_score,
            'f5Home':    f5_runs(home),
            'f5Away':    f5_runs(away),
            'status':    status,
        }
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
log = gist_fetch(f'mlb-log-{date_str}.json')
if not log:
    print(f'No mlb-log-{date_str}.json found — nothing to fill.')
    sys.exit(0)

# Only process entries that still have result=null
pending = [e for e in log if e.get('result') is None]
if not pending:
    print('All games already have results.')
    sys.exit(0)

espn = fetch_espn_results(date_str)
if not espn:
    print('No ESPN data returned.')
    sys.exit(0)

updated = 0
for entry in log:
    if entry.get('result') is not None:
        continue  # already filled

    key = f"{norm(entry['away'])}|{norm(entry['home'])}"
    r   = espn.get(key)
    if not r:
        # Try flipped norm (team name differences between Odds API and ESPN)
        for espn_key, espn_val in espn.items():
            if (norm(espn_val['homeName']) == norm(entry['home']) or
                norm(espn_val['awayName']) == norm(entry['away'])):
                r = espn_val
                break

    if not r or r['status'] != 'final':
        continue  # game not final yet

    home_score = r['homeScore']
    away_score = r['awayScore']
    f5_home    = r['f5Home']
    f5_away    = r['f5Away']
    total_runs = home_score + away_score

    # Determine winners
    winner = entry['home'] if home_score > away_score else entry['away']
    model_fav = entry['model']['rlFavTeam']  # model's predicted favorite

    # Did model's favored team cover -1.5?
    model_home_fav = (entry['model']['rlFavTeam'] == entry['home'])
    if model_home_fav:
        rl_result = 'cover' if (home_score - away_score) >= 2 else 'no_cover'
    else:
        rl_result = 'cover' if (away_score - home_score) >= 2 else 'no_cover'

    # Over/under result vs. Vegas total
    vegas_total = entry['vegas'].get('totalLine')
    ou_result   = None
    if vegas_total is not None:
        if total_runs > vegas_total:
            ou_result = 'over'
        elif total_runs < vegas_total:
            ou_result = 'under'
        else:
            ou_result = 'push'

    # F5 results
    f5_total      = (f5_home or 0) + (f5_away or 0)
    f5_vegas_line = entry['vegas'].get('f5TotalLine')
    f5_ou_result  = None
    if f5_vegas_line is not None and f5_home is not None:
        if f5_total > f5_vegas_line:
            f5_ou_result = 'over'
        elif f5_total < f5_vegas_line:
            f5_ou_result = 'under'
        else:
            f5_ou_result = 'push'

    f5_winner = None
    if f5_home is not None and f5_away is not None:
        if f5_home > f5_away:
            f5_winner = entry['home']
        elif f5_away > f5_home:
            f5_winner = entry['away']
        else:
            f5_winner = 'tie'

    # ML result: did model's predicted side win?
    model_home_win = entry['model']['homeWinProb'] >= entry['model']['awayWinProb']
    ml_correct = (winner == entry['home']) == model_home_win

    entry['result'] = {
        'homeScore':   home_score,
        'awayScore':   away_score,
        'winner':      winner,
        'totalRuns':   total_runs,
        'mlCorrect':   ml_correct,
        'rlResult':    rl_result,
        'ouResult':    ou_result,
        'f5HomeRuns':  f5_home,
        'f5AwayRuns':  f5_away,
        'f5TotalRuns': round(f5_total, 1) if f5_home is not None else None,
        'f5Winner':    f5_winner,
        'f5OuResult':  f5_ou_result,
    }
    updated += 1
    print(f'Filled: {entry["away"]} @ {entry["home"]} → {away_score}-{home_score} '
          f'(ML: {"✓" if ml_correct else "✗"}, RL: {rl_result}, O/U: {ou_result})')

if updated == 0:
    print('No completed games to fill yet.')
    sys.exit(0)

log_filename = f'mlb-log-{date_str}.json'
print(f'Patching {log_filename} ({updated} results filled)...')
try:
    status = gist_patch(log_filename, log)
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
