"""Throwaway recon #2 — answers what's needed to design the backtest harness:
  1. Do /games and /lines return a WHOLE SEASON in one call, or need per-week calls?
  2. When did lines go from composite/teamrankings to real sportsbook data?
  3. How far back does /player/returning go?
  4. Does CFBD rate FCS teams at all (for handling FBS-vs-FCS games consistently)?
  5. Exact shape of /games — is week/date/neutral-site/postseason all present?
Deleted once answered.
"""
import json, os, urllib.request, urllib.parse

KEY = os.environ['CFBD_API_KEY']
BASE = 'https://api.collegefootballdata.com'


def get(path, **params):
    url = f'{BASE}{path}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


print('=== 1a. /games — whole season in one call? ===')
for yr in (2015, 2024):
    try:
        g = get('/games', year=yr, seasonType='regular')
        weeks = sorted({x['week'] for x in g})
        print(f'  year {yr}: {len(g)} games, weeks present: {weeks[:3]}...{weeks[-3:]}')
        print('    sample:', json.dumps({k: g[0].get(k) for k in
              ('id','week','seasonType','startDate','neutralSite','homeTeam','awayTeam','homePoints','awayPoints','homeClassification','awayClassification')}, indent=1))
    except Exception as e:
        print(f'  year {yr} FAILED: {e}')

print('\n=== 1b. /lines — whole season in one call? ===')
for yr in (2015, 2024):
    try:
        l = get('/lines', year=yr, seasonType='regular')
        weeks = sorted({x['week'] for x in l})
        with_lines = sum(1 for x in l if x.get('lines'))
        print(f'  year {yr}: {len(l)} games, {with_lines} with lines, weeks: {weeks[:3]}...{weeks[-3:]}')
    except Exception as e:
        print(f'  year {yr} FAILED: {e}')

print('\n=== 2. When did lines become real-book (not composite)? ===')
for yr in (2014, 2016, 2018, 2020, 2021, 2022):
    try:
        l = get('/lines', year=yr, week=5, seasonType='regular')
        providers = {}
        for g in l:
            for ln in g.get('lines', []):
                providers[ln.get('provider')] = providers.get(ln.get('provider'), 0) + 1
        print(f'  {yr}: providers seen -> {providers}')
    except Exception as e:
        print(f'  {yr} FAILED: {e}')

print('\n=== 3. /player/returning — earliest year available? ===')
for yr in (2014, 2015, 2016, 2018, 2026):
    try:
        r = get('/player/returning', year=yr)
        print(f'  {yr}: {len(r)} rows')
    except Exception as e:
        print(f'  {yr} FAILED: {e}')

print('\n=== 4. Does /ratings/sp include FCS teams? ===')
try:
    sp = get('/ratings/sp', year=2024)
    confs = sorted({t.get('conference') for t in sp})
    print(f'  {len(sp)} teams, conferences represented: {confs}')
    fcs_like = [t['team'] for t in sp if t.get('conference') in
                ('Big Sky','Missouri Valley','Ivy','MEAC','Southland','SWAC','CAA','Southern')]
    print(f'  FCS-conference teams present: {fcs_like[:10]}')
except Exception as e:
    print(f'  FAILED: {e}')

print('\n=== 5. Season classification field on /games — lets us drop FCS-vs-FBS cleanly? ===')
try:
    g = get('/games', year=2024, week=1, seasonType='regular')
    sample = [x for x in g if x.get('homeClassification') != x.get('awayClassification')][:2]
    print('  mixed-classification games:', json.dumps([{k: s.get(k) for k in
          ('homeTeam','homeClassification','awayTeam','awayClassification')} for s in sample], indent=1))
except Exception as e:
    print(f'  FAILED: {e}')

print('\n=== 6. Rate limit remaining after this run ===')
try:
    req = urllib.request.Request(f'{BASE}/games?year=2024&week=1', headers={'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=15) as r:
        print('  ', {k: v for k, v in r.headers.items() if 'limit' in k.lower()})
except Exception as e:
    print('  FAILED:', e)
