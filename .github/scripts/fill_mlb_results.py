"""
fill_mlb_results.py
Runs each cycle after games start. For each entry in mlb-log-YYYY-MM-DD.json
that has result=null, looks up the final score and fills it in.
Also computes F5 runs from per-inning linescores.

Scores come from the MLB Stats API. This used to read ESPN's scoreboard, but
ESPN began returning 403 Forbidden to GitHub Actions runners around 2026-08-04,
which silently stopped every result from being filled (the script exits 0 on a
fetch failure, so the workflow step stayed green). statsapi.mlb.com is the
official feed, needs no key, and is already used by log_mlb_predictions.py from
the same runners.

Set BACKFILL_DAYS=N to also process the N-1 days before DATE, so a gap in
results heals itself on the next run instead of needing a manual backfill.
"""

import json, re, urllib.request, os, sys
from datetime import datetime, timezone, timedelta

import gist_api

# ── Env ───────────────────────────────────────────────────────────────────────
gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')
backfill = max(1, int(os.environ.get('BACKFILL_DAYS', '1')))

GIST_USER = 'loganthein'
STATS_URL = 'https://statsapi.mlb.com/api/v1/schedule'

# A game can reach abstractGameState 'Final' without having been played.
NOT_PLAYED = ('postponed', 'cancelled', 'canceled', 'suspended')

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
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data, indent=2), 'fill-mlb-results')

# ── Fetch final scores from the MLB Stats API ─────────────────────────────────
def fetch_game_results(date_str):
    """
    Returns dict: norm(away)|norm(home) → {
        homeScore, awayScore, homeName, awayName,
        f5Home, f5Away,   # runs over the first 5 innings (None if not available)
        status            # 'final' | 'in_progress' | 'scheduled'
    }
    """
    url = f'{STATS_URL}?sportId=1&date={date_str}&hydrate=linescore,team'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'fill-mlb-results'})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
    except Exception as e:
        print(f'MLB Stats API fetch failed: {e}', file=sys.stderr)
        return {}

    results = {}
    for day in d.get('dates', []):
        for game in day.get('games', []):
            state    = game.get('status', {}).get('abstractGameState', '')
            detailed = game.get('status', {}).get('detailedState', '').lower()

            if any(word in detailed for word in NOT_PLAYED):
                continue  # no score to record
            if state == 'Final':
                status = 'final'
            elif state == 'Live':
                status = 'in_progress'
            else:
                status = 'scheduled'

            teams = game.get('teams', {})
            home, away = teams.get('home', {}), teams.get('away', {})
            home_name = home.get('team', {}).get('name', '')
            away_name = away.get('team', {}).get('name', '')
            if not home_name or not away_name:
                continue

            # F5 needs five completed innings; a rain-shortened game has none.
            innings = game.get('linescore', {}).get('innings', [])
            if len(innings) >= 5:
                f5_home = sum(i.get('home', {}).get('runs', 0) or 0 for i in innings[:5])
                f5_away = sum(i.get('away', {}).get('runs', 0) or 0 for i in innings[:5])
            else:
                f5_home = f5_away = None

            key = f'{norm(away_name)}|{norm(home_name)}'
            results[key] = {
                'homeName':  home_name,
                'awayName':  away_name,
                'homeScore': int(home.get('score', 0) or 0),
                'awayScore': int(away.get('score', 0) or 0),
                'f5Home':    f5_home,
                'f5Away':    f5_away,
                'status':    status,
            }
    return results

# ── Fill one day ──────────────────────────────────────────────────────────────
def fill_date(date_str):
    """Fills any unscored entries for one date. Returns True on a clean pass."""
    log = gist_fetch(f'mlb-log-{date_str}.json')
    if not log:
        print(f'{date_str}: no log file — nothing to fill.')
        return True

    # Only process entries that still have result=null
    pending = [e for e in log if e.get('result') is None]
    if not pending:
        print(f'{date_str}: all games already have results.')
        return True

    scores = fetch_game_results(date_str)
    if not scores:
        print(f'{date_str}: no score data returned.')
        return False

    updated = 0
    for entry in log:
        if entry.get('result') is not None:
            continue  # already filled

        key = f"{norm(entry['away'])}|{norm(entry['home'])}"
        r   = scores.get(key)
        if not r:
            # Fall back to a single-side match (team naming differs between the
            # Odds API and the MLB Stats API, e.g. "Athletics" vs "Oakland Athletics")
            for cand in scores.values():
                if (norm(cand['homeName']) == norm(entry['home']) or
                    norm(cand['awayName']) == norm(entry['away'])):
                    r = cand
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
        print(f'{date_str}: no completed games to fill yet.')
        return True

    log_filename = f'mlb-log-{date_str}.json'
    print(f'Patching {log_filename} ({updated} results filled)...')
    try:
        status = gist_patch(log_filename, log)
        print(f'Gist PATCH status: {status}')
    except Exception as e:
        print(f'ERROR: Gist PATCH failed for {log_filename}: {e}', file=sys.stderr)
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
base  = datetime.strptime(date_str, '%Y-%m-%d')
dates = [(base - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(backfill)]

ok = True
for d in dates:
    if not fill_date(d):
        ok = False

sys.exit(0 if ok else 1)
