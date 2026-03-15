import json, urllib.request, os, sys

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ['DATE']
prefix   = os.environ.get('FILENAME_PREFIX', 'odds')

with open('/tmp/odds-raw.json') as f:
    content = f.read()

filename = f'{prefix}-{date_str}.json'
body = json.dumps({'files': {filename: {'content': content}}}).encode()

req = urllib.request.Request(
    f'https://api.github.com/gists/{gist_id}',
    data=body, method='PATCH',
    headers={
        'Authorization': f'token {gist_pat}',
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'fetch-odds-workflow',
    }
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print(f'Gist PATCH status: {r.status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
