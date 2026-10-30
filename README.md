# torvik-data

Bart Torvik blocks GitHub-hosted runners, so a daily Claude cloud session
(scheduled task "Daily Torvik fetch") runs `.github/scripts/fetch_torvik.py --out torvik`
from `main` and pushes `torvik/` here. `publish-torvik.yml` then copies those
files to the ratings Gist that the site reads. Only `torvik/` changes day to day.
