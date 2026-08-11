import json, urllib.request, os, sys

import gist_api

gist_id  = os.environ['GIST_ID']
gist_pat = os.environ['GIST_PAT']
date_str = os.environ['DATE']
prefix   = os.environ.get('FILENAME_PREFIX', 'odds')

with open('/tmp/odds-raw.json') as f:
    content = f.read()

filename = f'{prefix}-{date_str}.json'

try:
    status = gist_api.patch(gist_id, gist_pat, filename, content, 'fetch-odds-workflow')
    print(f'Gist PATCH status: {status}')
except Exception as e:
    print(f'ERROR: Gist PATCH failed: {e}', file=sys.stderr)
    sys.exit(1)
