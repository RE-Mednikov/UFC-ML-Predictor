# UFC-ML-Predictor

## UFCStats Scraper Pipeline

This repository now includes a polite UFCStats scraping pipeline that downloads and caches HTML locally, writes raw and cleaned tables to CSV and SQLite, and prepares the project for later snapshot generation.

### What gets scraped

- Completed events from the UFCStats completed events page.
- Fight pages for fight metadata and per-round statistics.
- Fighter profile pages for biographical data and profile validation.

### Output tables

Raw tables in `data/raw/` and `ufc_stats.db`:

- `events_raw`
- `fights_raw`
- `fight_stats_raw`
- `fighters_raw`

Clean tables in `data/clean/` and `ufc_stats.db`:

- `events_clean`
- `fighters_clean`
- `fights_clean`
- `fighter_fight_stats_clean`

### Polite scraping behavior

- Browser-like user agent.
- Retry logic for transient HTTP failures.
- Request timeout.
- Sleep between requests.
- Local HTML cache in `data/raw_html/`.
- Resumable runs that skip already-scraped rows unless `--force` is used.

### Quick validation run

Scrape only 2 completed events first:

```bash
python -m src.scrape.scrape_events --limit 2
python -m src.scrape.scrape_event_fights
python -m src.scrape.scrape_fight_details
python -m src.scrape.scrape_fighters
python -m src.clean.build_clean_tables
python -m src.validation.validation_report
```

### Validation report

`src.validation.validation_report` writes:

- `validation_report.csv`
- `validation_summary.txt`

It exits non-zero on critical integrity failures.

### Next modeling step

The next step will be to sort fights chronologically, generate point-in-time fighter snapshots before each fight, and then create one row per fight with diff features such as:

- `age_diff`
- `height_diff`
- `reach_diff`
- `ufc_fights_diff`
- `ufc_win_rate_diff`
- `days_since_last_fight_diff`
- `SLpM_diff`
- `SApM_diff`
- `strike_differential_diff`
- `TD_avg_diff`
- `TD_accuracy_diff`
- `TD_defense_diff`
- `TD_attempt_rate_diff`
- `control_time_per_fight_diff`
- `sub_avg_diff`
- `overall_elo_diff`
- `ufc_minutes_diff`
- `last3_win_rate_diff`
- `last3_SLpM_diff`
- `last3_TD_avg_diff`
- `finish_rate_diff`
- `sig_strike_accuracy_diff`
- `sig_strike_defense_diff`
- `sig_strike_attempted_per_min_diff`