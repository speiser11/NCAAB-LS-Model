"""
build_cfb_backtest.py
Backtests the from-scratch CFB rating engine (cfb_rating_engine.py) against
real historical closing lines, and reports whether disagreeing with the
market predicts anything about who actually covers.

Why a from-scratch engine instead of backtesting FPI or CFBD's SP+ directly:
both return today's value when you ask for a past season, even at a specific
past week — verified by pulling the same team's rating at three different
weeks of a finished season and getting an identical number back each time.
Backtesting against a rating that already knows how the season ended isn't
measuring a model, it's measuring hindsight. This engine only ever uses
information that existed before the game it is rating.

Train/test split — 2015-2020 to fit, 2021-2024 to evaluate — was picked from
what the data actually supports, not arbitrarily: CFBD's historical lines
before ~2021 are composite/aggregator numbers (teamrankings, numberfire),
while 2021 onward has real single-book coverage (William Hill, Caesars,
Bovada, and DraftKings by 2024). So:
  - TRAIN seasons fit the engine's free parameters (K, home-field, season
    carryover, returning-production weight, margin cap) by minimizing
    walk-forward prediction error against ACTUAL GAME MARGINS ONLY. No market
    data is used in fitting, at all — this half of the design has no lookahead
    concern even in principle, because every prediction is made from
    information strictly before that game.
  - TEST seasons take the resulting FIXED engine and compare its projections
    against real market lines it was never fit against, on games from seasons
    the parameters never saw. Two independent guardrails, not one.

Usage:
  export CFBD_API_KEY=your_key_here
  python3 optimize/build_cfb_backtest.py

Output: a summary report to stdout. Pass --dump-csv PATH to also write one row
per evaluated game (our projection, the market line, the actual result) for
deeper inspection.
"""

import json, os, sys, time, urllib.request, urllib.parse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from cfb_rating_engine import RatingEngine, EngineParams

CFBD_API_KEY = os.environ.get('CFBD_API_KEY')
if not CFBD_API_KEY:
    sys.exit('CFBD_API_KEY not set. export CFBD_API_KEY=your_key_here')

BASE = 'https://api.collegefootballdata.com'
TRAIN_SEASONS = list(range(2015, 2021))   # fit params: actual outcomes only
TEST_SEASONS  = list(range(2021, 2025))   # evaluate: real market lines
SEED_SP_YEAR  = TRAIN_SEASONS[0] - 1      # 2014 — one-time cold start only

# Preference order for picking "the" market line out of several providers per
# game. Consensus first (most representative of "the market" as a whole);
# real single books next; composite/aggregator sources last, since they are
# themselves a blend rather than a tradeable price.
LINE_PREFERENCE = [
    'consensus', 'DraftKings', 'William Hill (New Jersey)', 'Caesars Sportsbook (Colorado)',
    'Caesars', 'Bovada', 'teamrankings', 'numberfire',
]


def cfbd_get(path, **params):
    url = f'{BASE}{path}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {CFBD_API_KEY}'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def pick_line(lines):
    if not lines:
        return None
    by_provider = {ln.get('provider'): ln for ln in lines}
    for name in LINE_PREFERENCE:
        if name in by_provider and by_provider[name].get('spread') is not None:
            return by_provider[name]['spread']
    for ln in lines:
        if ln.get('spread') is not None:
            return ln['spread']
    return None


# ── Load one season's worth of games, joined to lines where available ────────
def load_season(year):
    print(f'  fetching {year}...', file=sys.stderr)
    games = cfbd_get('/games', year=year, seasonType='regular')
    lines_raw = cfbd_get('/lines', year=year, seasonType='regular')
    lines_by_id = {g['id']: pick_line(g.get('lines')) for g in lines_raw}

    out = []
    for g in games:
        if g.get('homeClassification') != 'fbs' or g.get('awayClassification') != 'fbs':
            continue   # engine only ever rates FBS-vs-FBS games — see module docstring
        if g.get('homePoints') is None or g.get('awayPoints') is None:
            continue   # not yet played / no result on file
        out.append({
            'id':     g['id'],
            'date':   g['startDate'],
            'home':   g['homeTeam'],
            'away':   g['awayTeam'],
            'homeScore': g['homePoints'],
            'awayScore': g['awayPoints'],
            'neutral':   bool(g.get('neutralSite')),
            'marketSpread': lines_by_id.get(g['id']),
        })
    out.sort(key=lambda x: x['date'])
    return out


def load_returning_production(year):
    try:
        rows = cfbd_get('/player/returning', year=year)
    except Exception as e:
        print(f'  returning-production fetch failed for {year} (non-fatal): {e}', file=sys.stderr)
        return {}
    # percentPPA: fraction of last season's total Predicted Points Added that
    # is walking back onto the field this season — a single, comparable number
    # per team regardless of roster size, and safely preseason (it describes
    # departures/returns, not anything from this season's games).
    return {r['team']: r.get('percentPPA') or 0.0 for r in rows}


def load_seed_ratings(year):
    """One-time cold start for the very first train season, from a completed
    prior season's SP+. Used exactly once — every later season carries over
    this engine's own rating instead."""
    try:
        sp = cfbd_get('/ratings/sp', year=year)
    except Exception as e:
        print(f'  seed SP+ fetch failed for {year}: {e}', file=sys.stderr)
        return {}
    return {r['team']: r['rating'] for r in sp if r.get('rating') is not None}


# ── Walk one season forward through the engine ────────────────────────────────
def run_season(engine, season_games, returning_prod, record_eval, results):
    # Chronological order is the one property this whole backtest depends on —
    # every prediction must be made from strictly-before information. load_season
    # already sorts, but this function enforces it too rather than trusting the
    # caller: a future caller passing unsorted games would otherwise silently
    # violate the no-lookahead guarantee with no visible error. Caught by a unit
    # test that fed shuffled input and got different (wrong) predictions back.
    season_games = sorted(season_games, key=lambda g: g['date'])
    teams = sorted({g['home'] for g in season_games} | {g['away'] for g in season_games})
    engine.new_season(teams, returning_prod)
    for g in season_games:
        pre_pred = engine.predict_home_margin(g['home'], g['away'], g['neutral'])
        if record_eval and g['marketSpread'] is not None:
            actual_margin = g['homeScore'] - g['awayScore']
            our_home_spread = -pre_pred          # same sign convention as marketSpread
            results.append({
                'date': g['date'], 'home': g['home'], 'away': g['away'],
                'ourSpread': our_home_spread, 'marketSpread': g['marketSpread'],
                'actualMargin': actual_margin,
            })
        engine.update(g['home'], g['away'], g['homeScore'], g['awayScore'], g['neutral'])


# ── Fit params on TRAIN seasons: minimize walk-forward MAE vs actual margins,
# no market data involved at any point in this function ──────────────────────
def train_mae(train_data, params, seed_ratings):
    engine = RatingEngine(params)
    engine.seed(seed_ratings)
    total_err, n = 0.0, 0
    for year in TRAIN_SEASONS:
        games, ret_prod = train_data[year]
        teams = sorted({g['home'] for g in games} | {g['away'] for g in games})
        engine.new_season(teams, ret_prod)
        for g in games:
            pred = engine.predict_home_margin(g['home'], g['away'], g['neutral'])
            actual = g['homeScore'] - g['awayScore']
            total_err += abs(actual - pred)
            n += 1
            engine.update(g['home'], g['away'], g['homeScore'], g['awayScore'], g['neutral'])
    return total_err / n if n else float('inf')


def fit_params(train_data, seed_ratings):
    # Modest grid — this is pure computation over already-fetched data, so grid
    # size costs runtime, not API calls. Widen it later if the optimum sits at
    # an edge of the current ranges.
    grid = {
        'k':              [0.05, 0.08, 0.12, 0.16, 0.20],
        'hfa':            [1.5, 2.0, 2.5, 3.0, 3.5],
        'carryover':      [0.5, 0.65, 0.8],
        'returning_coef': [0.0, 10.0, 20.0, 30.0],
        'margin_cap':     [None, 21.0, 28.0],
    }
    best, best_mae = None, float('inf')
    combos = 0
    for k in grid['k']:
        for hfa in grid['hfa']:
            for carry in grid['carryover']:
                for rc in grid['returning_coef']:
                    for cap in grid['margin_cap']:
                        combos += 1
                        p = EngineParams(k=k, hfa=hfa, carryover=carry,
                                          returning_coef=rc, margin_cap=cap)
                        mae = train_mae(train_data, p, seed_ratings)
                        if mae < best_mae:
                            best_mae, best = mae, p
    print(f'  grid search: {combos} combinations evaluated', file=sys.stderr)
    return best, best_mae


# ── ATS grading, same convention as fill_cfb_results.py ──────────────────────
def grade(our_spread, market_spread, actual_margin, take_edge, lean_edge):
    edge = our_spread - market_spread
    abs_edge = abs(edge)
    if abs_edge < lean_edge:
        return None, abs_edge
    took_home = edge < 0   # our model likes the home side relative to market
    line = market_spread if took_home else -market_spread
    covered = (actual_margin if took_home else -actual_margin) + line
    result = 'W' if covered > 0 else 'L' if covered < 0 else 'push'
    tier = 'take' if abs_edge >= take_edge else 'lean'
    return (tier, result), abs_edge


def main():
    print(f'Loading TRAIN seasons {TRAIN_SEASONS} (outcomes only, no market data used to fit)...')
    train_data = {}
    for yr in TRAIN_SEASONS:
        games = load_season(yr)
        ret = load_returning_production(yr)
        train_data[yr] = (games, ret)
        print(f'    {yr}: {len(games)} FBS-vs-FBS games')

    seed_ratings = load_seed_ratings(SEED_SP_YEAR)
    print(f'  seed ratings from {SEED_SP_YEAR} SP+: {len(seed_ratings)} teams (used once, cold start only)')

    print('\nFitting engine parameters via grid search on TRAIN seasons...')
    best_params, best_mae = fit_params(train_data, seed_ratings)
    print(f'  best: k={best_params.k} hfa={best_params.hfa} carryover={best_params.carryover} '
          f'returning_coef={best_params.returning_coef} margin_cap={best_params.margin_cap}')
    print(f'  walk-forward MAE on TRAIN seasons (vs actual margins, no market involved): {best_mae:.2f} pts')

    print(f'\nLoading TEST seasons {TEST_SEASONS} (real market lines, never used in fitting)...')
    engine = RatingEngine(best_params)
    engine.seed(seed_ratings)
    # Replay TRAIN seasons once more with the fitted params so the engine
    # carries real history into the test window, rather than starting test
    # evaluation from a cold seed.
    for yr in TRAIN_SEASONS:
        games, ret = train_data[yr]
        run_season(engine, games, ret, record_eval=False, results=[])

    eval_rows = []
    for yr in TEST_SEASONS:
        games = load_season(yr)
        ret = load_returning_production(yr)
        with_line = sum(1 for g in games if g['marketSpread'] is not None)
        print(f'    {yr}: {len(games)} FBS-vs-FBS games, {with_line} with a market line')
        run_season(engine, games, ret, record_eval=True, results=eval_rows)

    # ── Evaluation ────────────────────────────────────────────────────────────
    # ourSpread and marketSpread are both "home spread" convention (negative =
    # home favored); actualMargin is home_score - away_score. A perfect
    # spread equals -actualMargin, so compare on that basis explicitly.
    our_err  = [abs(-r['ourSpread'] - r['actualMargin']) for r in eval_rows]
    mkt_err  = [abs(-r['marketSpread'] - r['actualMargin']) for r in eval_rows]
    n = len(eval_rows)
    print(f'\n=== Out-of-sample evaluation: {n} games, {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} ===')
    print(f'  our MAE vs actual margin:    {sum(our_err)/n:.2f} pts')
    print(f'  market MAE vs actual margin: {sum(mkt_err)/n:.2f} pts')
    print(f'  (our model losing to the market here is expected — the market has line movement,')
    print(f'   injury news, and weather that this engine does not see. The question is not')
    print(f"   whether we beat the market's accuracy, it's whether OUR DISAGREEMENTS with it")
    print(f'   predict anything about who covers.)')

    buckets = defaultdict(lambda: {'W': 0, 'L': 0, 'push': 0})
    for r in eval_rows:
        graded, abs_edge = grade(r['ourSpread'], r['marketSpread'], r['actualMargin'],
                                  take_edge=6.0, lean_edge=3.0)
        if not graded:
            continue
        tier, result = graded
        buckets[tier][result] += 1

    print('\n=== ATS by disagreement size (out-of-sample, never used in fitting) ===')
    print(f"  {'tier':<6} {'W':>4} {'L':>4} {'push':>5} {'win%':>7}  (52.4% needed to beat -110)")
    for tier in ('lean', 'take'):
        b = buckets[tier]
        decided = b['W'] + b['L']
        pct = (b['W'] / decided * 100) if decided else float('nan')
        print(f"  {tier:<6} {b['W']:>4} {b['L']:>4} {b['push']:>5} {pct:>6.1f}%")

    if '--dump-csv' in sys.argv:
        path = sys.argv[sys.argv.index('--dump-csv') + 1]
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['date','home','away','ourSpread','marketSpread','actualMargin'])
            w.writeheader()
            w.writerows(eval_rows)
        print(f'\nWrote {len(eval_rows)} rows to {path}')


if __name__ == '__main__':
    main()
