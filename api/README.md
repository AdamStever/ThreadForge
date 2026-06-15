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

# set an API key (see Authentication), then run the API against that database:
dotnet user-secrets set "Auth:ApiKey" "dev-key" --project api/ThreadForge.Api
dotnet run --project api/ThreadForge.Api --urls http://localhost:5099
```

Protected endpoints require an `X-API-Key` header — see [Authentication](#authentication).
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

## Authentication

Every endpoint except the open liveness paths (`/`, `/health`) requires an
`X-API-Key` header matching the configured key. Requests without a valid key get
`401`. Auth **fails closed**: if no key is configured, protected endpoints are
rejected rather than left open.

The key is read from configuration (`Auth:ApiKey`) and is **never** stored in
the repo — `appsettings.json` ships only an empty placeholder. Supply the real
key out-of-band:

```bash
# development — user-secrets (stored in your user profile, not the repo)
dotnet user-secrets set "Auth:ApiKey" "<your-key>" --project api/ThreadForge.Api

# production — environment variable (double underscore maps to Auth:ApiKey)
FeatureStore__DbPath=/path/threadforge.db Auth__ApiKey=<your-key> dotnet run --project api/ThreadForge.Api
```

```bash
curl http://localhost:5099/health                                   # open
curl -H "X-API-Key: <your-key>" http://localhost:5099/runs
curl -H "X-API-Key: <your-key>" http://localhost:5099/runs/1
curl -H "X-API-Key: <your-key>" http://localhost:5099/runs/1/signals/volatility
```

The key comparison is constant-time to avoid leaking it through timing.

## Build

```bash
dotnet build api/ThreadForge.slnx
```
