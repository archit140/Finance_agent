# CentrAlign Worker Demo

## Start

```powershell
uv run uvicorn sandbox_api:app --port 8000
uv run streamlit run app.py
```

Open `http://localhost:8501` and select **CentrAlign worker**.

## 1. Happy path

Reset the ledger, enter `Enter Acme invoice`, and run the worker. Show the plan, SOP, API actions, evidence, and verified result.

## 2. Human approval

Reset the ledger, enter `Enter Northwind invoice`, and run with approval unchecked. Show the policy pause. Check approval and run again. Show the verified ledger result.

## 3. Recovery

From a terminal, reset the ledger and inject one failure:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/reset
Invoke-RestMethod -Method Post http://127.0.0.1:8000/faults/next-ledger-write
```

Run `Enter Acme invoice` in the UI. Show the first `503`, the second successful attempt, and final verification.

## 4. Safe failure

Run `Enter Globex invoice`. Show that the worker does not write an unrelated invoice and reports that no matching invoice was found.

## 5. Evaluation and audit evidence

```powershell
uv run python evaluate_worker.py
Get-Content data/worker_traces.jsonl -Tail 5
```

Use `worker_eval_results.json` and the audit traces in the README or demo recording.
