"""
fill_cfb_results.py
Runs each cycle after games start. For each entry in cfb-log-YYYY-MM-DD.json
with result=null, finds a final score on ESPN and grades the model's pick
against it — mirrors fill_mlb_results.py's role for MLB.

Grading is ATS (against the spread), since that is what the model's edge is
measured in. A game with no pick (tier is null — either side unrated, or the
model's disagreement with the market was large enough to distrust) still gets
its final score recorded, so the log carries a complete picture of the week
even where the model stayed silent.

ESPN quirk this script works around: cdn.espn.com's scoreboard endpoint takes a
`dates` parameter but silently ignores it — tested against three different
dates and it returned identical current-week data every time. Using it here
would have filled nothing and never said why. The schedule endpoint genuinely
scopes by year/week (verified: a real past week returned 49/49 games with their
final score intact), so results come from there instead: fetch the last few
weeks once, index every game by its calendar date, then fill against that
index rather than making one ESPN call per date.

Walks back BACKFILL_DAYS (default 10) so a gap — an API outage, a missed run —
heals itself once the source recovers, same as the MLB version.
"""

import json, os, re, sys, urllib.request, unicodedata
from datetime import datetime, timedelta, timezone

import gist_api

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ.get('DATE') or datetime.now(timezone.utc).strftime('%Y-%m-%d')
backfill = max(1, int(os.environ.get('BACKFILL_DAYS', '1')))

GIST_USER          = 'loganthein'
ESPN_CURRENT_URL   = 'https://cdn.espn.com/core/college-football/scoreboard?xhr=1&groups=80'
ESPN_SCHEDULE_URL  = 'https://cdn.espn.com/core/college-football/schedule?xhr=1&year={year}&week={week}&group=80'
WEEKS_BACK         = 3   # current week + this many prior weeks, comfortably covers a 10-day backfill


def _base(s):
    s = unicodedata.normalize('NFD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', s)).strip()


def norm(s):
    return _base(s).replace(' ', '')


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
    return gist_api.patch(gist_id, gist_pat, filename, json.dumps(data, indent=2), 'fill-cfb-results')


def espn_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def build_results_index():
    """
    Returns dict: 'YYYY-MM-DD' -> { 'norm(away)|norm(home)' -> {homeName, awayName,
    homeScore, awayScore, status} }, covering the current week and WEEKS_BACK
    prior weeks.
    """
    try:
        cur = espn_get(ESPN_CURRENT_URL)
        sbd = cur.get('content', {}).get('sbData', {})
        year, week = sbd['season']['year'], sbd['week']['number']
    except Exception as e:
        print(f'Could not determine current week: {e}', file=sys.stderr)
        return {}

    index = {}
    for wk in range(max(1, week - WEEKS_BACK), week + 1):
        try:
            sched = espn_get(ESPN_SCHEDULE_URL.format(year=year, week=wk))
        except Exception as e:
            print(f'Note: schedule fetch failed for week {wk}: {e}')
            continue
        for day_key, blk in (sched.get('content', {}).get('schedule', {}) or {}).items():
            date_iso = f'{day_key[:4]}-{day_key[4:6]}-{day_key[6:8]}'
            bucket = index.setdefault(date_iso, {})
            for g in blk.get('games', []):
                comp = (g.get('competitions') or [{}])[0]
                state = comp.get('status', {}).get('type', {}).get('name', '')
                status = 'final' if state == 'STATUS_FINAL' else \
                         'in_progress' if state == 'STATUS_IN_PROGRESS' else 'scheduled'
                cs = comp.get('competitors', [])
                home = next((c for c in cs if c.get('homeAway') == 'home'), None)
                away = next((c for c in cs if c.get('homeAway') == 'away'), None)
                if not home or not away:
                    continue
                h_name = home.get('team', {}).get('displayName', '')
                a_name = away.get('team', {}).get('displayName', '')
                bucket[f'{norm(a_name)}|{norm(h_name)}'] = {
                    'homeName':  h_name,
                    'awayName':  a_name,
                    'homeScore': int(home.get('score', 0) or 0),
                    'awayScore': int(away.get('score', 0) or 0),
                    'status':    status,
                }
    return index


def grade_pick(entry, home_score, away_score):
    """Returns 'W' | 'L' | 'push' | None (no pick was made on this game)."""
    m = entry.get('model') or {}
    pick = m.get('pick')
    if not pick:
        return None
    margin = home_score - away_score  # positive = home won by this much
    took_home = pick['side'] == entry['home']
    line = pick['line']
    covered = (margin + line) if took_home else (-margin + line)
    if covered > 0:
        return 'W'
    if covered < 0:
        return 'L'
    return 'push'


def fill_date(date_str, results_by_date):
    log = gist_fetch(f'cfb-log-{date_str}.json')
    if not log:
        print(f'{date_str}: no log file — nothing to fill.')
        return True

    pending = [e for e in log if e.get('result') is None]
    if not pending:
        print(f'{date_str}: all games already have results.')
        return True

    day_results = results_by_date.get(date_str, {})
    if not day_results:
        print(f'{date_str}: no ESPN data for this date (outside the fetched week window, or none scheduled).')
        return True   # not an error — just nothing to grade yet

    updated = 0
    for entry in log:
        if entry.get('result') is not None:
            continue

        key = f"{norm(entry['away'])}|{norm(entry['home'])}"
        r = day_results.get(key)
        if not r:
            for cand in day_results.values():
                if (norm(cand['homeName']) == norm(entry['home']) or
                    norm(cand['awayName']) == norm(entry['away'])):
                    r = cand
                    break
        if not r or r['status'] != 'final':
            continue

        home_score, away_score = r['homeScore'], r['awayScore']
        winner = entry['home'] if home_score > away_score else \
                 entry['away'] if away_score > home_score else None
        entry['result'] = {
            'homeScore':  home_score,
            'awayScore':  away_score,
            'winner':     winner,
            'atsResult':  grade_pick(entry, home_score, away_score),
        }
        updated += 1
        tag = entry['result']['atsResult'] or 'no pick'
        print(f'Filled: {entry["away"]} @ {entry["home"]} -> {away_score}-{home_score}  ({tag})')

    if updated == 0:
        print(f'{date_str}: no completed games to fill yet.')
        return True

    filename = f'cfb-log-{date_str}.json'
    print(f'Patching {filename} ({updated} result(s) filled)...')
    try:
        status = gist_patch(filename, log)
        print(f'Gist PATCH status: {status}')
    except Exception as e:
        print(f'ERROR: Gist PATCH failed for {filename}: {e}', file=sys.stderr)
        return False
    return True


results_by_date = build_results_index()
print(f'Indexed {sum(len(v) for v in results_by_date.values())} games '
      f'across {len(results_by_date)} date(s) from ESPN.')

base = datetime.strptime(date_str, '%Y-%m-%d')
dates = [(base - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(backfill)]

ok = True
for d in dates:
    if not fill_date(d, results_by_date):
        ok = False

sys.exit(0 if ok else 1)
