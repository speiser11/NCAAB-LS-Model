"""Throwaway reconnaissance script. Answers, from real CFBD data:
  1. Is SP+ genuinely as-of-week, or does /ratings/sp only ever return final?
  2. How deep does /lines go, and what book/market coverage does a real week have?
  3. Does /ppa/teams (efficiency) vary week to week within one season?
  4. What does /player/returning look like — is it safely pre-season?
Deleted once answered. Prints only aggregates and small samples — nothing secret.
"""
import json, os, urllib.request, urllib.parse

KEY = os.environ['CFBD_API_KEY']
BASE = 'https://api.collegefootballdata.com'


def get(path, **params):
    url = f'{BASE}{path}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


print('=== 1. SP+ ratings: is there a week-level endpoint, or only season-end? ===')
try:
    sp_season = get('/ratings/sp', year=2024)
    print(f'/ratings/sp?year=2024 -> {len(sp_season)} teams, sample:', json.dumps(sp_season[0], indent=1)[:400])
except Exception as e:
    print('  /ratings/sp failed:', e)

try:
    sp_week = get('/ratings/sp', year=2024, team='Ohio State')
    print('\n/ratings/sp?year=2024&team=Ohio State ->', json.dumps(sp_week, indent=1)[:500])
except Exception as e:
    print('  team-filtered SP+ failed:', e)

print('\n=== 2. Historical lines: depth + schema ===')
for yr in (2013, 2019, 2024):
    try:
        lines = get('/lines', year=yr, week=5, seasonType='regular')
        n = len(lines)
        with_line = sum(1 for g in lines if g.get('lines'))
        print(f'  year {yr} week 5: {n} games, {with_line} with a lines[] entry')
        if lines and lines[0].get('lines'):
            print('    sample game:', json.dumps({k: lines[0][k] for k in ('homeTeam','awayTeam','homeScore','awayScore')}))
            print('    sample line entry:', json.dumps(lines[0]['lines'][0], indent=1)[:400])
    except Exception as e:
        print(f'  year {yr}: FAILED {e}')

print('\n=== 3. PPA (efficiency) — does it change week to week? ===')
try:
    for wk in (3, 8, 13):
        ppa = get('/ppa/teams', year=2024, week=wk, team='Ohio State')
        print(f'  week {wk}:', json.dumps(ppa, indent=1)[:300] if ppa else 'empty')
except Exception as e:
    print('  /ppa/teams failed:', e)

print('\n=== 4. Returning production — preseason, one row per team-season? ===')
try:
    ret = get('/player/returning', year=2024)
    print(f'  {len(ret)} rows, sample:', json.dumps(ret[0], indent=1)[:400] if ret else 'empty')
except Exception as e:
    print('  /player/returning failed:', e)

print('\n=== 5. Rate limit headroom (informational) ===')
try:
    req = urllib.request.Request(f'{BASE}/ratings/sp?year=2024', headers={'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=15) as r:
        print('  headers of interest:', {k: v for k, v in r.headers.items() if 'ratelimit' in k.lower() or 'limit' in k.lower()})
except Exception as e:
    print('  rate-limit probe failed:', e)
