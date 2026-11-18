# Round agent runbook

You drive exactly one round of the electronic-structure harness loop, then stop.
A round advances every session in `loop.yaml` by one generation: each open
prompt is answered by its own fresh Claude Opus 5.5 subagent, then the harness
scores the answers and writes the next prompts. Another agent monitors the loop
read-only while you work; do not wait for it or talk to it.

Work in `/workspace` on the branch named in `loop.yaml` (`branch:`). Push only
to that branch. Do not open, update, or merge pull requests.

## Rules

- Do not edit harness files: `esm_harness/`, `configs/`, `ablations/`,
  `baselines/`, `benchmarks/`, `tests/`, `pyproject.toml`, `loop.yaml`,
  `LOOP.md`, `README.md`. The lock records their fingerprint and `round end`
  refuses to score if any changed. If the harness looks broken, stop and report
  it; do not fix it.
- Never write or edit a response, meta, prompt, program, or trajectory file
  yourself. Only subagents write responses; only the harness commands write the
  rest.
- One subagent per launch, launched fresh. Never resume a subagent and never
  give one subagent more than one candidate.
- Launch with exactly the model each request names (`model`), never another.
- Log every launch with `meta`, failures included. Do not discard a response a
  subagent wrote.
- Run one round only. When `round end` succeeds, push and finish.

## Steps

1. Sync and install:

   ```bash
   cd /workspace
   BRANCH=$(python3 -c "import yaml;print(yaml.safe_load(open('loop.yaml'))['branch'])")
   git fetch origin "$BRANCH" && git checkout -B "$BRANCH" "origin/$BRANCH"
   python3 -m pip install -q -e '.[dev]'
   ```

2. Find your own agent id: call the `cursor-cloud` tool `run-info` and take its
   `bcId` (`bc-...`). Call it `ME` below.

3. Open the round and publish the lock so the monitor sees it:

   ```bash
   python3 -m esm_harness round begin --agent-id "$ME"
   git add -A LOOP_LOCK.json trajectories && git commit -qm "round: begin" && git push -q origin "HEAD:$BRANCH"
   ```

   If it exits saying the loop is paused, that the round is held by another
   agent, that every session is done, that it is waiting for cluster
   evaluations, or that the cluster cannot take jobs, report that and stop
   without committing or pushing anything. Never take over another agent's
   round.

4. The output lists `requests`. For each one launch a subagent with the Task
   tool. Put all launches in a single message so they run in parallel:
   - `subagent_type`: `generalPurpose`
   - `model`: the request's `model`
   - `prompt`: the request's `instructions`, verbatim
   - `description`: `Opus 5.5 <candidate_id>`
   - `run_in_background`: true when the model ends in `-max`, otherwise false

5. After each subagent returns, record it with its agent id (`cloudAgentBcId`
   from the Task result):
   - It wrote the response file:
     `python3 -m esm_harness meta --cid <candidate_id> --model <model> --agent-id <id>`
   - It failed (timeout, error, blocked, or no response file):
     `python3 -m esm_harness meta --cid <candidate_id> --model <model> --agent-id <id> --error "<reason>"`

   Then run `python3 -m esm_harness round requests`. Any request still listed
   needs a new fresh subagent (step 4 again for those only). The harness stops
   listing a candidate after three logged failures.

6. When `round requests` prints `[]`, close the round:

   ```bash
   python3 -m esm_harness round end --agent-id "$ME"
   git add -A LOOP_LOCK.json trajectories && git commit -qm "round: end" && git push -q origin "HEAD:$BRANCH"
   ```

   If the push is rejected, `git pull --rebase origin "$BRANCH"` and push again.

   Sessions that evaluate on the cluster are not scored here: `round end`
   submits one cluster task per molecule, and the session's status becomes
   `evaluating`. A later round's `round begin` collects the results, scores
   them, and only then writes the next prompts.

7. Finish with a short report: the round number, each candidate's id, agent id,
   status, fitness, and time ratio (from the `round end` output), every
   candidate still `evaluating` with its cluster task ids, and every failed
   launch with its reason.
