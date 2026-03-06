"""
Scrape KenPom official (referee) stats and push to the GitHub Gist
alongside torvik-ratings.json as a new file 'kenpom-refs.json'.

Environment variables required:
  KENPOM_USER   — your KenPom login email
  KENPOM_PASS   — your KenPom password
  GITHUB_TOKEN  — GitHub PAT with gist write scope
  GIST_ID       — ID of the target Gist (same one used for Torvik data)

Usage (local):
  set KENPOM_USER=you@email.com
  set KENPOM_PASS=yourpassword
  set GITHUB_TOKEN=ghp_xxx
  set GIST_ID=44dd1f1464e3d7e2689b25ba758d4ea9
  py optimize/scrape_refs.py
"""

import os
import json
import sys
import requests
from bs4 import BeautifulSoup

KENPOM_BASE  = 'https://kenpom.com'
LOGIN_URL    = f'{KENPOM_BASE}/user/login/'
OFFICIALS_URL = f'{KENPOM_BASE}/officials.php'


def kenpom_session(email: str, password: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Load login page to capture any CSRF/hidden fields
    r = s.get(LOGIN_URL, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    form = soup.find('form')
    form_data = {'email': email, 'password': password}

    # Carry any hidden input fields (CSRF tokens, etc.)
    if form:
        for inp in form.find_all('input', {'type': 'hidden'}):
            if inp.get('name'):
                form_data[inp['name']] = inp.get('value', '')

    r2 = s.post(LOGIN_URL, data=form_data, timeout=15)
    r2.raise_for_status()

    # Verify login succeeded by checking for logout link
    if 'logout' not in r2.text.lower() and 'sign out' not in r2.text.lower():
        # Some KenPom setups redirect to home on success — check cookies
        if not any('kenpom' in c.lower() or 'session' in c.lower() for c in s.cookies.keys()):
            raise RuntimeError(
                'KenPom login appears to have failed. '
                'Check KENPOM_USER / KENPOM_PASS environment variables.'
            )
    return s


def scrape_officials(session: requests.Session) -> dict:
    r = session.get(OFFICIALS_URL, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    # KenPom uses a DataTable with id="officials-table" or similar
    table = soup.find('table', {'id': 'officials-table'}) or soup.find('table')
    if not table:
        raise RuntimeError('Could not find officials table on kenpom.com/officials.php')

    thead = table.find('thead')
    tbody = table.find('tbody')
    if not thead or not tbody:
        raise RuntimeError('Officials table missing thead or tbody')

    headers = [th.get_text(strip=True) for th in thead.find_all('th')]
    print(f'  Columns found: {headers}')

    # Column name mappings (KenPom may rename these between seasons)
    COL_ALIASES = {
        'official':  ['Official', 'Name', 'Referee'],
        'games':     ['Games', 'G', 'GP'],
        'foulRate':  ['Foul Rate', 'Fouls/40', 'Fouls Per 40', 'Foul Rate/40'],
        'ftRate':    ['FT Rate', 'Adj FT Rate', 'FTRate'],
        'conf':      ['Conf', 'Conference'],
    }

    def find_col(aliases):
        for alias in aliases:
            for i, h in enumerate(headers):
                if h.strip().lower() == alias.lower():
                    return i
        return None

    idx = {k: find_col(v) for k, v in COL_ALIASES.items()}
    print(f'  Column indices: {idx}')

    if idx['official'] is None:
        raise RuntimeError(f'Cannot find official name column in headers: {headers}')

    refs = {}
    for row in tbody.find_all('tr'):
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cells:
            continue

        name = cells[idx['official']] if idx['official'] < len(cells) else ''
        if not name:
            continue

        entry = {'name': name}
        if idx['games'] is not None and idx['games'] < len(cells):
            try: entry['games'] = int(cells[idx['games']])
            except ValueError: pass
        if idx['foulRate'] is not None and idx['foulRate'] < len(cells):
            try: entry['foulRate'] = float(cells[idx['foulRate']])
            except ValueError: pass
        if idx['ftRate'] is not None and idx['ftRate'] < len(cells):
            try: entry['ftRate'] = float(cells[idx['ftRate']])
            except ValueError: pass
        if idx['conf'] is not None and idx['conf'] < len(cells):
            entry['conf'] = cells[idx['conf']]

        refs[name.lower()] = entry  # keyed by lowercase name for JS lookup

    return refs


def push_to_gist(token: str, gist_id: str, content: str):
    url = f'https://api.github.com/gists/{gist_id}'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    payload = {'files': {'kenpom-refs.json': {'content': content}}}
    r = requests.patch(url, json=payload, headers=headers, timeout=15)
    r.raise_for_status()
    print(f'  Pushed to Gist: {r.json()["html_url"]}')


def main():
    email  = os.environ.get('KENPOM_USER', '')
    pw     = os.environ.get('KENPOM_PASS', '')
    token  = os.environ.get('GITHUB_TOKEN', '')
    gist_id = os.environ.get('GIST_ID', '44dd1f1464e3d7e2689b25ba758d4ea9')

    if not email or not pw:
        print('ERROR: Set KENPOM_USER and KENPOM_PASS environment variables.')
        sys.exit(1)
    if not token:
        print('ERROR: Set GITHUB_TOKEN environment variable.')
        sys.exit(1)

    print('Logging into KenPom...')
    session = kenpom_session(email, pw)
    print('  Logged in successfully.')

    print('Scraping officials.php...')
    refs = scrape_officials(session)
    print(f'  Found {len(refs)} referees.')

    if not refs:
        print('ERROR: No referees parsed — check table structure.')
        sys.exit(1)

    # Preview a few entries
    for k, v in list(refs.items())[:3]:
        print(f'  {k}: {v}')

    payload = json.dumps(refs, indent=2)
    print(f'\nPushing kenpom-refs.json to Gist {gist_id}...')
    push_to_gist(token, gist_id, payload)
    print('Done.')


if __name__ == '__main__':
    main()
