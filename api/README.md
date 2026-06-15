# ThreadForge.Api

A small C#/.NET (ASP.NET Core minimal API) service that exposes the detection
results over HTTP. It is a **read-only serving layer over the shared feature
store**: the Python pipeline detects and writes (`run_detection.py --store`),
this reads that same SQLite database and serves it. The API never mutates the
store (it opens the database read-only).

```
Python pipeline ──writes──► SQLite feature store ──reads──► ThreadForge.Api ──HTTP──► clients
```

## Requirements

- .NET SDK 10+ (`dotnet --version`)

## Run

```bash
# from the repo root, first produce a store with the Python side:
python scripts/run_detection.py data/raw/ec2_cpu_utilization_5f5533.csv --store threadforge.db

# then run the API, pointing it at that database:
dotnet run --project api/ThreadForge.Api --urls http://localhost:5099
```

The database path defaults to `threadforge.db` (relative to the working
directory). Override it with configuration:

```bash
# environment variable (note the double underscore)
FeatureStore__DbPath=/path/to/threadforge.db dotnet run --project api/ThreadForge.Api
```

or in `appsettings.json`:

```json
{ "FeatureStore": { "DbPath": "threadforge.db" } }
```

## Endpoints

| Method & path | Returns |
|---|---|
| `GET /health` | `{ "status": "ok" }` |
| `GET /runs` | all recorded runs |
| `GET /runs/{id}` | one run's summary (404 if unknown) |
| `GET /runs/{id}/signals` | the signal names recorded for the run |
| `GET /runs/{id}/signals/{name}` | one signal's time series |
| `GET /runs/{id}/stream` | the raw input stream |

Signal/stream/list endpoints accept an optional `?channel=<name>` query
parameter (default `value`) for multi-channel stores.

```bash
curl http://localhost:5099/runs
curl http://localhost:5099/runs/1
curl http://localhost:5099/runs/1/signals/volatility
```

## Build & test

```bash
dotnet build api/ThreadForge.slnx
```

Authentication and secrets management are intentionally not here yet — they land
in a follow-up.
