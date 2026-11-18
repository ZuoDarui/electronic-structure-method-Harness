# Driver audit

Each session's `calls.jsonl`, `meta/` and `trajectory.jsonl` checked against the platform transcripts of the driver and of every subagent it launched. Details are in each session's `audit.json`.

| run | bound model | candidates | launches | responses verified | call_failed | violations | notes |
|---|---|---|---|---|---|---|---|
| cell-00-hf_minimal-iterative-w0.5-high | claude-opus-5-5-high | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-01-hf_spec-iterative-w0.5-high | claude-opus-5-5-high | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-02-hf_show_baseline-iterative-w0.5-high | claude-opus-5-5-high | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-03-hf_minimal-best_of_n-w0.5-high | claude-opus-5-5-high | 12 | 12 | 12 | 0 | 0 | 0 |
| cell-04-hf_minimal-iterative-w0.0-high | claude-opus-5-5-high | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-05-hf_minimal-iterative-w2.0-high | claude-opus-5-5-high | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-06-hf_minimal-iterative-w0.5-medium | claude-opus-5-5-medium | 4 | 4 | 4 | 0 | 0 | 0 |
| cell-07-hf_minimal-iterative-w0.5-max | claude-opus-5-5-max | 4 | 9 | 3 | 1 | 0 | 1 |
| hf-opus55 | claude-opus-5-5-high | 8 | 8 | 8 | 0 | 0 | 3 |

## Findings

- note, `cell-07-hf_minimal-iterative-w0.5-max-g03-s0`: agent bc-7c77f328-cee1-59b7-be7e-bb8b65891259 made a search_replace call that has no recorded result (a failed write, retried)
- note, `hf-opus55-g00-s0`: agent bc-455ed491-9b0c-5f64-b7e7-c077e85db8de made a search_replace call that has no recorded result (a failed write, retried)
- note, `hf-opus55-g00-s0`: agent bc-455ed491-9b0c-5f64-b7e7-c077e85db8de made a search_replace call that has no recorded result (a failed write, retried)
- note, `hf-opus55-g00-s0`: agent bc-455ed491-9b0c-5f64-b7e7-c077e85db8de read its own response file before writing it
- note, `hf-opus55-g00-s0`: agent bc-df5d8b05-ba16-5885-a616-afa42147f46a was launched before this session was created (an earlier run with the same id) and is not counted
