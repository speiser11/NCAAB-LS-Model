# MLB Model — Code Review & Roadmap

## What You Built (Great Work)

The framework is solid. Here's what's in place and working:

### Data Pipeline
- **ESPN MLB API** — game schedules, scores, status (live, final, pre)
- **MLB Stats API** — probable pitcher stats: ERA, FIP, WHIP, K/9, BB/9, handedness
- **The Odds API via GitHub Gist** — moneylines + run lines, updated every 30 min via GitHub Actions
- **Static park factors** — all 30 parks with run-scoring multipliers
- **Static team offense (wRC+)** — vsL / vsR splits per team *(updated to 2025 actuals — see below)*

### Model
- **Pitcher quality score** — `(4.50 - FIP) / 1.0`, baseline 4.50 = league average. Good foundation.
- **Offensive adjustment** — `(wRC+ - 100) / 20` scaled to unit contributions per side
- **Run total projection** — `9.0 - pitcherQuality + offenseAdj * 2.0`, scaled by park factor, clamped 5.5–14.0
- **Win probability** — logistic curve on projected run differential, home team gets +2.5% base
- **Vegas edge** — model win prob vs. implied market prob from moneyline
- **Run line model** — separate probability for covering ±1.5

### UI
- Game cards with projected total, model moneyline, Vegas edge
- Detail modal: full pitcher breakdown, park factor, run projection breakdown, run line comparison
- Date picker, status filter (live / final / upcoming)

---

## What We Fixed for Opening Day

1. **ESPN season type** — was set to `seasontype=3` (Spring Training). Flipped to `seasontype=2` (Regular Season).
2. **Broken odds loader** — a stub `loadMLBOdds()` at the bottom of the file was silently overriding the real implementation. Removed the stub; odds now load correctly from the Gist.
3. **wRC+ data** — updated from 2024 to **2025 FanGraphs full-season actuals** (vsL/vsR for all 30 teams). No public source publishes 2026 preseason team-level splits, so 2025 actuals are the best available proxy. Notable shifts: Yankees and Mets up significantly, Guardians and Rangers down, Tigers have a strong reverse split (114 vsL / 98 vsR).

---

## Gaps & Priorities

### 🔴 High Priority

**1. No bullpen data**
The model only accounts for the starting pitcher. Bullpens contribute ~4–6 runs per game (roughly half the total). A bad bullpen on a short-staffed team is a massive edge the model currently ignores entirely. There's already a `TODO` comment in the code for this.

*Plan:* Fetch season bullpen ERA/FIP from MLB Stats API by team, blend with starter quality. Could weight by save situation likelihood (close games late = bullpen matters more).

**2. No recent form**
The offense ratings are season averages. A team that's gone 2-10 over the last two weeks looks the same as a team that's 10-2. The NCAAB model has a `FORM_WEIGHT` parameter for exactly this — blending season average with recent performance delta.

*Plan:* Use MLB Stats API to fetch last-N-games team stats, compute a rolling wRC+ or OPS, blend with static season average (e.g., 70% season / 30% last 15 games).

**3. No rest / back-to-back tracking**
In the NCAAB model, a back-to-back game applies a -1.8 pt penalty (calibrated from 16k games). MLB has similar effects — starting pitchers on short rest, bullpens taxed from previous night, travel across time zones.

*Plan:* Fetch last 7 days of ESPN schedule per team to detect B2B situations. Apply a small run-line penalty (~0.3–0.5 runs) to taxed teams. Calibrate once we have backtest data.

---

### 🟡 Medium Priority

**4. Injury data**
Missing a star lineup player is huge; missing a bench player is nothing. The NCAAB model weights injuries by minutes-per-game. MLB equivalent is plate appearances or WAR.

*Plan:* ESPN injury API or roster page. Flag teams missing high-PA players, apply a small offensive penalty. Lower priority than bullpen/form.

**5. Home/away splits**
Most teams have a 5–10 pt wRC+ swing between home and road. Currently all offense ratings are neutral. Would improve accuracy at extreme parks (Coors, Petco, Oracle).

---

### 🟢 Lower Priority

**6. Weather**
Wind direction and speed materially affects run scoring at outdoor parks. Easy to fetch, small average impact — but meaningful for Coors, Wrigley on windy days.

**7. Line movement tracking**
Logging opening line vs. current line tells you if sharp money has moved the number. Would help identify edges the model finds but the market has already priced in.

---

## Longer-Term: Backtesting

This is where the NCAAB model really matured. The architecture should be:

1. **Log predictions daily** — for each game, store: model total, model win prob, Vegas line, and eventual outcome
2. **After ~4–6 weeks of data** — run a simple regression: which inputs (FIP weight, offense weight, park factor, form delta) best predict actual outcomes?
3. **Tune weights** — the model has hardcoded values like `9.0` baseline, `2.0` offense multiplier, `1.2` logistic scale. These should be empirically calibrated, not guessed.
4. **Track ATS and O/U performance** — are we beating the market on totals? On moneylines? Which game types (divisional, interleague, dome vs. outdoor) is the model strongest on?

The NCAAB model started with the same approach and the backtesting revealed some surprising things — certain factors we thought would matter barely moved accuracy, while rest/travel data we almost skipped turned out to be significant.

---

## Architecture Reference (NCAAB vs. MLB)

| Signal | NCAAB | MLB (current) | MLB (goal) |
|---|---|---|---|
| Base ratings | Torvik adjO/adjD (daily Gist) | Static 2025 wRC+ | Rolling wRC+ via GitHub Action |
| Pitcher/defense | Torvik adjD | FIP/ERA (live API) | FIP + bullpen ERA |
| Recent form | 14-day Torvik delta | ❌ None | Last 15 games blend |
| Rest/B2B | ✅ -1.8 pt penalty | ❌ None | ~-0.4 run penalty |
| Injuries | ✅ MPG-weighted | ❌ None | PA-weighted |
| Park/venue | ✅ Home court +X pts | ✅ Park factors | ✅ Already good |
| Odds | ✅ Gist, 30 min | ✅ Gist, 30 min | ✅ Already good |
| Backtesting | ✅ Calibrated | ❌ None | Log → regress → tune |

---

## Suggested Sprint Order

1. **Sprint 7** — Bullpen data: fetch team bullpen ERA/FIP, integrate into run total model
2. **Sprint 8** — Prediction logging: store daily model outputs to a Gist or Firestore so we have backtest data accumulating from Opening Day
3. **Sprint 9** — Recent form: rolling team offense from MLB Stats API
4. **Sprint 10** — Rest/B2B tracking
5. **Sprint 11** — First backtest pass: calibrate baseline (9.0), offense multiplier (2.0), logistic scale (1.2)
6. **Sprint 12** — Injury flags
