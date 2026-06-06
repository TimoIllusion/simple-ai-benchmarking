# Testing live log streaming locally

Exercise the dashboard's live-console streaming end-to-end on localhost — the
same path a RunPod pod uses with `saib-runpod --debug`, but with **no RunPod and
no Docker**. (Docker, to simulate a real pod environment, can be layered on later;
this setup deliberately stays host-local.)

```
tools/dummy_workload.py  ->  logfile  ->  saib-logship  ->  local dashboard DB
                                                   |
                                  browser live console  +  read-endpoint verify
```

The pieces:

| Piece | Repo | Role |
|-------|------|------|
| `tools/serve_local.sh` | ai-benchmark-database | migrate + mint a token + run the dev server |
| `tools/dummy_workload.py` | simple-ai-benchmarking | fake benchmark console (stdout/stderr, sleeps) |
| `saib-logship` | simple-ai-benchmarking | tails the logfile, ships lines to the DB |
| `tools/local_stream_demo.py` | simple-ai-benchmarking | drives the workload + shipper, then verifies |

## One command (auto-boots the server)

From the `simple-ai-benchmarking` repo, with the DB repo checked out as a sibling:

```bash
tools/local_stream_demo.py --start-server --db-repo ../ai-benchmark-database
```

It boots the dashboard, streams a dummy run, prints the streamed console as the
**server** returns it, and prints `PASS`/`FAIL`. Open the printed
`/dashboard/runs/<run_id>/` URL while it runs to watch the live tail. Add `--fail`
to confirm a failed run is reported as `failed`.

## Two terminals (watch a real server)

```bash
# terminal 1 — ai-benchmark-database:
tools/serve_local.sh                      # prints AI_BENCHMARK_DATABASE_TOKEN=...

# terminal 2 — simple-ai-benchmarking:
AI_BENCHMARK_DATABASE_TOKEN=<token> tools/local_stream_demo.py
```

Then browse to `http://127.0.0.1:8000/dashboard/` (run table, with a "live
console" link per run) or directly to the printed run URL.

## What this proves

- `saib-logship` tails a growing file and ships only complete lines, in order.
- The ingest endpoint stores them and the read endpoint serves them by cursor
  (the same polling the browser console uses).
- Run lifecycle is reported (`running` → `completed`/`failed`).

The dummy workload merges stderr into stdout just like the pod's
`exec > >(tee …) 2>&1`, so the streamed console matches what `--debug` produces
on a real pod.
