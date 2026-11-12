# Roblox Up-and-Coming Forecast

This entry forecasts the top 10 games on the Roblox "Up and Coming" chart for the next 72 hours, using historical chart snapshots.

## What it does

The program reads past chart history and predicts which games will occupy the top 10 positions 72 hours from now. It combines several components:

- **Rank diffusion**: models how game ranks naturally drift over time.
- **Empirical transition**: uses observed rank-to-rank transitions from the history.
- **Playing-spike surge**: boosts games that show a sudden increase in player count.
- **Low-rank floor**: prevents games with the lowest expected ranks from dropping too far.
- **Logistic tilt**: a runtime-fitted curve that adjusts probabilities based on the history itself.
- **Dead-zone rows**: games ranked 31–50 in the last chart are assigned zero probability of reaching the top 10.

The result is a ranked list of the most likely top 10 games.

## Inputs

Located in the `input/` directory:

- `history.ndjson` — historical chart snapshots, one JSON object per line.
- `task.json` — task configuration, including the target horizon (72 hours).

## Output

Written to the `output/` directory:

- `forecast.json` — the predicted top 10 games for the next 72 hours.

## How to run

