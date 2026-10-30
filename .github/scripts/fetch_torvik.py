"""
fetch_torvik.py
Fetches Bart Torvik's team ratings and pushes them to the ratings Gist as
torvik-ratings.json (+ a dated copy and torvik-meta.json), which index.html
and the NCAAB pick log read.

Torvik answers GitHub-hosted runners with 403 Forbidden (it started in May
2026 and was total by October), so this runs somewhere Torvik doesn't block.
Standard library only, so it runs on any machine with Python 3.

Two ways to deliver the result:
  --out DIR   write torvik-ratings.json, torvik-meta.json and the dated copy
              into DIR. The daily Claude cloud session uses this and pushes
              DIR to the torvik-data branch, where publish-torvik.yml copies
              it to the Gist (cloud sessions can't write Gists themselves).
  (default)   PATCH the files straight to the Gist; needs GIST_TOKEN, a PAT
              with gist scope for the speiser11 ratings Gist.
"""

import argparse, http.cookiejar, json, os, sys, urllib.parse, urllib.request
from datetime import date, datetime, timezone

GIST_ID = '44dd1f1464e3d7e2689b25ba758d4ea9'

ap = argparse.ArgumentParser()
ap.add_argument('--out', help='write the files into this directory instead of the Gist')
args = ap.parse_args()

# Season year: from October on, it's next spring's season (Nov 2026 → 2027).
today = date.today()
YEAR = today.year + 1 if today.month >= 10 else today.year

opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
opener.addheaders = [
    ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
    ('Referer', 'https://barttorvik.com/'),
]

def get(url, data=None):
    """Returns (status, body text); HTTP errors come back as a status, not an exception."""
    body = urllib.parse.urlencode(data).encode() if data else None
    try:
        with opener.open(url, body, timeout=30) as r:
            return r.status, r.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        return e.code, ''

# In season: trank.php serves JSON after a bot check — a GET sets the cookie,
# then a POST imitating the page's auto-submitting form returns the data.
url = f'https://barttorvik.com/trank.php?json=1&year={YEAR}'
get(url)
status, text = get(url, {'js_test_submitted': '1'})
print(f'trank.php: HTTP {status}')
try:
    parsed = json.loads(text) if status == 200 else None
    content, source = text, 'trank'
except ValueError:
    parsed = None

# Before the season's first game trank.php redirects to the preseason page
# instead. Fall back to the preseason ratings in {YEAR}_team_results.json,
# reshaped into the trank row layout the site parses:
# [team, adjOE, adjDE, barthag, record, wins, games, ..., adjT at index 15].
if parsed is None:
    status, text = get(f'https://barttorvik.com/{YEAR}_team_results.json')
    print(f'{YEAR}_team_results.json: HTTP {status}')
    if status != 200:
        sys.exit(f'ERROR: Torvik returned HTTP {status}' +
                 (' — this machine is blocked too' if status == 403 else ''))
    parsed = [[row[1], row[4], row[6], row[8], row[3], 0, 0] + [0] * 8 + [row[-1]]
              for row in json.loads(text)]
    content, source = json.dumps(parsed), 'preseason'

if not isinstance(parsed, list) or len(parsed) < 100:
    sys.exit(f'ERROR: unexpected Torvik response: {text[:200]}')
print(f'Fetched {len(parsed)} teams from Torvik (season {YEAR}, {source})')

meta = {
    'updated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'season': YEAR,
    'team_count': len(parsed),
    'source': source,
}
files = {
    'torvik-ratings.json':               {'content': content},
    f'torvik-{today.isoformat()}.json':  {'content': content},
    'torvik-meta.json':                  {'content': json.dumps(meta)},
}

if args.out:
    os.makedirs(args.out, exist_ok=True)
    for name, f in files.items():
        with open(os.path.join(args.out, name), 'w', encoding='utf-8') as fh:
            fh.write(f['content'])
    print(f'Wrote {len(files)} files to {args.out}')
    sys.exit(0)

token = os.environ.get('GIST_TOKEN')
if not token:
    sys.exit('ERROR: GIST_TOKEN is not set')
req = urllib.request.Request(
    f'https://api.github.com/gists/{GIST_ID}',
    data=json.dumps({'files': files}).encode('utf-8'),
    method='PATCH',
    headers={
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github+json',
        'Content-Type': 'application/json',
        'User-Agent': 'fetch-torvik',
    },
)
with urllib.request.urlopen(req, timeout=30) as r:
    print(f'Gist updated: HTTP {r.status} — season {YEAR}, {source}')
