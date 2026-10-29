# Publish the Python Experiments

**Goal:** Publish the latest local Python experiments on the repository's main branch,
with legacy notebooks archived and the current reward-hacking results easy to find.

**Scope:** Integrate the existing refactor and optional GPU-kernel branches, including
the completed data-night analysis files currently uncommitted in the refactor worktree.
Keep source worktrees and their generated model artifacts in place.

## Tasks

- [x] Integrate both development histories on a temporary integration branch; carry over
  the seven pending analysis/source/test documents and their report figures.
- [x] Refresh the root README, explain the notebook archive, publish a concise English
  reward-hacking case study, and exclude agent workspaces and model artifacts.
- [x] Run the full CPU suite, validate notebook preservation and document links, and review
  the publication contents. Deliver the reviewed snapshot by fast-forwarding main,
  pushing without force, and verifying the remote SHA.

## Verification and review focus

- The four legacy notebooks must retain their original Git blob contents; Windows
  checkout line-ending conversion is not a notebook-content change.
- Commands documented for the fixed configuration must match the actual CLI/configs.
- Reported metrics must distinguish training slices, full test results, and subset sweeps.
- No model weights, checkpoints, environment secrets, or nested worktrees may be published.
- Optional GPU features must stay opt-in; CPU tests must pass without a training stack.

This is repository integration and documentation work. Existing experiment implementations
are preserved; no new training run is required or claimed. The user's request authorizes
the final merge and push. Keep the existing developer worktrees intact.

Integration finding: the optional kernel tests imported PyTorch unconditionally,
so the documented CPU-only developer setup failed at collection. The initial full
suite reproduced both errors; dependency-aware skips were added to those existing
test modules before rerunning the suite. No kernel implementation was changed.

Verification: 114 tests passed, 4 skipped for unavailable optional dependencies;
all four notebook Git blobs match the original main revision; refreshed documentation
links resolve; ten shell scripts passed `bash -n`; the Python wheel built successfully.
A fresh read-only publication review found no Critical or Important issues. Its
historical-parser caveat was added to README. GPU numerical/performance behavior was
not rerun; this integration preserves the existing opt-in implementation and reports.
