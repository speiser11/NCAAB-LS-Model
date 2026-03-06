"""
Scrape KenPom official (referee) stats AND team ratings, push both to the
GitHub Gist as 'kenpom-refs.json' and 'kenpom-teams.json'.

NOTE: KenPom blocks GitHub Actions IP ranges (403 Forbidden on login page).
Run this locally from your machine — KenPom doesn't block residential IPs.

Environment variables required:
  KENPOM_USER   — your KenPom login email
  KENPOM_PASS   — your KenPom password
  GITHUB_TOKEN  — GitHub PAT with gist write scope (same token used for Torvik)
  GIST_ID       — ID of the target Gist (defaults to the Torvik gist)

Usage (Windows, one-time or via Task Scheduler):
  set KENPOM_USER=you@email.com
  set KENPOM_PASS=yourpassword
  set GITHUB_TOKEN=ghp_xxx
  py optimize/scrape_refs.py

Run weekly — both refs and team ratings update throughout the season.
The app falls back gracefully when either file is absent.
"""

import os
import re
import json
import sys
import requests
import cloudscraper
from urllib.parse import urljoin
from bs4 import BeautifulSoup

KENPOM_BASE   = 'https://kenpom.com'
LOGIN_URL     = f'{KENPOM_BASE}/user/login/'
OFFICIALS_URL = f'{KENPOM_BASE}/officials.php'
TEAMS_URL     = f'{KENPOM_BASE}/index.php'


def kenpom_session(email: str, password: str) -> requests.Session:
    # cloudscraper handles Cloudflare's JS challenge automatically
    s = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})

    # Visit homepage to get cookies and find the login form action URL
    r0 = s.get(KENPOM_BASE + '/', timeout=15)
    soup0 = BeautifulSoup(r0.text, 'html.parser')

    # Find the login form — try by id, then by action pattern, then first form
    form = (soup0.find('form', {'id': 'login-form'})
            or soup0.find('form', action=lambda a: a and 'login' in a.lower())
            or soup0.find('form'))

    if form:
        action = form.get('action', '')
        login_post_url = action if action.startswith('http') else urljoin(KENPOM_BASE + '/', action)
        print(f'  Found login form, action: {login_post_url}')
    else:
        # Fall back to known URL patterns
        login_post_url = KENPOM_BASE + '/user/login_post.php'
        print(f'  No login form found on homepage, trying: {login_post_url}')

    form_data = {}
    if form:
        for inp in form.find_all('input'):
            if inp.get('name'):
                form_data[inp['name']] = inp.get('value', '')

    form_data['email']    = email
    form_data['password'] = password

    r2 = s.post(login_post_url, data=form_data, timeout=15)
    r2.raise_for_status()

    # Accept login if: response contains logout link, OR we got a session cookie,
    # OR the handler returned a redirect back to the homepage (common pattern)
    logged_in = (
        'logout' in r2.text.lower()
        or 'sign out' in r2.text.lower()
        or any('session' in c.lower() or 'auth' in c.lower() or 'user' in c.lower()
               for c in s.cookies.keys())
        or r2.url == KENPOM_BASE + '/'
        or r2.url == KENPOM_BASE
    )
    if not logged_in:
        raise RuntimeError(
            'KenPom login appears to have failed. '
            'Check KENPOM_USER / KENPOM_PASS environment variables.'
        )
    return s


def scrape_officials(session: requests.Session) -> dict:
    r = session.get(OFFICIALS_URL, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    table = soup.find('table', {'id': 'officials-table'}) or soup.find('table')
    if not table:
        raise RuntimeError('Could not find officials table on kenpom.com/officials.php')

    # Collect all rows — KenPom may not use <thead>/<tbody>
    all_rows = table.find_all('tr')
    if not all_rows:
        raise RuntimeError('Officials table has no rows')

    # First row with <th> cells = header row; otherwise use first <tr>
    header_row = next((r for r in all_rows if r.find('th')), all_rows[0])
    headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
    print(f'  Columns found: {headers}')

    # Data rows = all rows that have <td> cells
    data_rows = [r for r in all_rows if r.find('td')]

    # Column name mappings — exact match first, no partial fallback to avoid mismatches
    COL_ALIASES = {
        'official': ['NameFAA', 'Official', 'Name', 'Referee'],
        'rating':   ['Rating'],
        'games':    ['Gms', 'Games', 'GP'],
        'lastGame': ['Last Game'],
    }

    def find_col(aliases):
        for alias in aliases:
            for i, h in enumerate(headers):
                if h.strip().lower() == alias.strip().lower():
                    return i
        return None

    idx = {k: find_col(v) for k, v in COL_ALIASES.items()}
    print(f'  Column indices: {idx}')

    if idx['official'] is None:
        raise RuntimeError(f'Cannot find official name column. Headers were: {headers}')

    refs = {}
    for row in data_rows:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if not cells or len(cells) < 2:
            continue

        raw_name = cells[idx['official']] if idx['official'] < len(cells) else ''
        if not raw_name:
            continue

        # KenPom appends the bias rating to the name e.g. "Kipp Kissinger-0.6"
        # Extract it and clean the name
        bias_match = re.search(r'([+-]\d+\.\d+)$', raw_name)
        name  = raw_name[:bias_match.start()].strip() if bias_match else raw_name.strip()
        bias  = float(bias_match.group(1)) if bias_match else None

        entry = {'name': name}
        if bias is not None:
            entry['bias'] = bias   # +ve = ref favors home, -ve = favors away
        if idx['rating'] is not None and idx['rating'] < len(cells):
            try: entry['rating'] = float(cells[idx['rating']])
            except ValueError: pass
        if idx['games'] is not None and idx['games'] < len(cells):
            try: entry['games'] = int(float(cells[idx['games']]))
            except ValueError: pass
        if idx['lastGame'] is not None and idx['lastGame'] < len(cells):
            entry['lastGame'] = cells[idx['lastGame']]

        refs[name.lower()] = entry  # keyed by clean lowercase name for JS lookup

    return refs


def norm_team_py(name: str) -> str:
    """Mirror of JS normTeam() — produces the same keys used in the model."""
    s = name.lower()
    s = re.sub(r'\s*\([^)]{0,6}\)\s*', ' ', s)   # strip (NY), (FL) etc.
    s = re.sub(r"[^a-z0-9\s'&.-]", '', s)
    s = re.sub(r'([a-z])\.([a-z])\.', r'\1\2', s) # n.c. → nc
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r' st\.$', ' state', s)
    s = re.sub(r'\bst\b', 'state', s)
    return s


def scrape_teams(session: requests.Session) -> dict:
    r = session.get(TEAMS_URL, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    table = soup.find('table', {'id': 'ratings-table'}) or soup.find('table')
    if not table:
        raise RuntimeError('Could not find ratings table on kenpom.com')

    all_rows = table.find_all('tr')
    if not all_rows:
        raise RuntimeError('Ratings table has no rows')

    # KenPom uses a two-row thead: row 1 = group labels, row 2 = column names.
    # Pick the <th>-containing row with the most non-empty header text.
    th_rows = [r for r in all_rows if r.find('th')]
    if not th_rows:
        raise RuntimeError('No header rows found in ratings table')
    header_row = max(th_rows, key=lambda r: sum(1 for c in r.find_all(['th', 'td']) if c.get_text(strip=True)))
    headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
    print(f'  Team table columns: {headers}')

    COL_ALIASES = {
        'rank':  ['Rk', 'Rank'],
        'team':  ['Team'],
        'adjEM': ['AdjEM'],
        'adjO':  ['AdjOE', 'AdjO', 'Off. Efficiency'],
        'adjD':  ['AdjDE', 'AdjD', 'Def. Efficiency'],
        'adjT':  ['AdjT', 'Tempo'],
        'luck':  ['Luck'],
    }

    def find_col(aliases):
        for alias in aliases:
            for i, h in enumerate(headers):
                if h.strip().lower() == alias.strip().lower():
                    return i
        return None

    idx = {k: find_col(v) for k, v in COL_ALIASES.items()}
    print(f'  Column indices: {idx}')

    if idx['team'] is None:
        raise RuntimeError(f'Cannot find team column. Headers: {headers}')

    def cell_text(td):
        """Return only direct text of the <td>, skipping child element text (e.g. rank spans)."""
        from bs4 import NavigableString
        return ''.join(str(t) for t in td.children if isinstance(t, NavigableString)).strip()

    data_rows = [r for r in all_rows if r.find('td')]
    teams = {}
    for row in data_rows:
        cells = row.find_all('td')
        if not cells or len(cells) < 3:
            continue

        team_td = cells[idx['team']] if idx['team'] < len(cells) else None
        if not team_td:
            continue
        team_name = team_td.get_text(strip=True)
        if not team_name:
            continue

        key = norm_team_py(team_name)
        entry = {'name': team_name}

        for field in ['rank', 'adjEM', 'adjO', 'adjD', 'adjT', 'luck']:
            col_i = idx.get(field)
            if col_i is not None and col_i < len(cells):
                raw = cell_text(cells[col_i])
                if not raw:
                    raw = cells[col_i].get_text(strip=True)
                try:
                    val = float(raw)
                    entry[field] = int(val) if field == 'rank' else round(val, 4)
                except ValueError:
                    pass

        teams[key] = entry

    return teams


def push_to_gist(token: str, gist_id: str, content: str, filename: str = 'kenpom-refs.json'):
    url = f'https://api.github.com/gists/{gist_id}'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    payload = {'files': {filename: {'content': content}}}
    r = requests.patch(url, json=payload, headers=headers, timeout=15)
    r.raise_for_status()
    print(f'  Pushed {filename} to Gist: {r.json()["html_url"]}')


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

    for k, v in list(refs.items())[:3]:
        print(f'  {k}: {v}')

    payload = json.dumps(refs, indent=2)
    print(f'\nPushing kenpom-refs.json to Gist {gist_id}...')
    push_to_gist(token, gist_id, payload, 'kenpom-refs.json')

    print('\nScraping index.php (team ratings + luck)...')
    teams = scrape_teams(session)
    print(f'  Found {len(teams)} teams.')

    if not teams:
        print('ERROR: No teams parsed — check table structure.')
        sys.exit(1)

    for k, v in list(teams.items())[:3]:
        print(f'  {k}: {v}')

    payload = json.dumps(teams, indent=2)
    print(f'\nPushing kenpom-teams.json to Gist {gist_id}...')
    push_to_gist(token, gist_id, payload, 'kenpom-teams.json')
    print('Done.')


if __name__ == '__main__':
    main()
