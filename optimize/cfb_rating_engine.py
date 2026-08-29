"""
cfb_rating_engine.py
The rating system itself, kept separate from data-fetching so its math can be
unit-tested with synthetic games and no network access at all.

Why this exists instead of using FPI or CFBD's SP+ for a backtest: both are
season-end numbers even when queried for a past season and a specific week —
verified by pulling the same team's rating at three different weeks of a
finished season and getting an identical value back both times (see the CFBD
and ESPN reconnaissance in this project's history). Backtesting against a
rating that already knows how the season ended isn't measuring a model, it's
measuring hindsight.

This engine only ever uses information that existed before the game it is
rating: a team's rating going into a game reflects last season's carried-over
value, this season's returning production (known before kickoff), and this
season's actual results so far — nothing from the game itself or later.

Rating scale is points of expected margin on a neutral field, same convention
as FPI: rating(A) - rating(B) + HFA (if A is home) = A's expected margin.
"""

from dataclasses import dataclass, field


@dataclass
class EngineParams:
    k: float = 0.1              # update speed — how much one game moves a rating
    hfa: float = 2.5            # home-field advantage, in points
    carryover: float = 0.65     # fraction of last season's rating kept season-to-season
    returning_coef: float = 0.0 # points of preseason rating per unit of returning production
    margin_cap: float | None = 24.0  # cap on the error term a single game can contribute
    default_rating: float = 0.0      # for a team with no history at all (new FBS entrant)


class RatingEngine:
    def __init__(self, params: EngineParams):
        self.p = params
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.p.default_rating)

    def new_season(self, teams: list[str], returning_production: dict[str, float]):
        """
        Called once at the start of each season, before any games are processed.
        Blends each team's carried-over rating with this season's returning
        production. A team with no prior-season rating (new to FBS, or the very
        first season this engine has ever seen) starts from default_rating,
        not from returning production alone — returning production only ever
        adjusts an existing prior, it is not itself a full rating.
        """
        updated = {}
        for team in teams:
            prev = self.ratings.get(team, self.p.default_rating)
            rp = returning_production.get(team, 0.0)
            updated[team] = self.p.carryover * prev + self.p.returning_coef * rp
        self.ratings = updated

    def seed(self, ratings: dict[str, float]):
        """Used once, for the very first season, to seed from an external prior
        (e.g. a completed prior season's published rating) rather than starting
        every team at zero with no information at all."""
        self.ratings = dict(ratings)

    def predict_home_margin(self, home: str, away: str, neutral: bool) -> float:
        hfa = 0.0 if neutral else self.p.hfa
        return self.get(home) - self.get(away) + hfa

    def update(self, home: str, away: str, home_score: int, away_score: int, neutral: bool):
        """One game's worth of learning. Must be called in chronological order —
        the engine has no notion of time itself, the caller owns that guarantee."""
        actual_margin = home_score - away_score
        expected_margin = self.predict_home_margin(home, away, neutral)
        error = actual_margin - expected_margin
        if self.p.margin_cap is not None:
            error = max(-self.p.margin_cap, min(self.p.margin_cap, error))
        step = self.p.k * error
        self.ratings[home] = self.get(home) + step
        self.ratings[away] = self.get(away) - step
