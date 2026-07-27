# Expert artifact provenance

The benchmark contains byte-for-byte copies of the twelve publication reference
solutions from the parent repository's `tasks/task-01/solution/solution_1.py`
through `tasks/task-12/solution/solution_12.py` at ORBIT-Q commit
`0d8208f782965d62857d4f62befc3f493656c1eb`. The immutable copies live under
`references/task-XX/`.

Each `src/solutions/task-XX/solution_N.py` was initialized as a second
byte-for-byte copy of its reference. These are the only solution files intended
for normal autoresearch edits.

The copied evaluators preserve the original computation, timer boundary,
validity checks, and output contract. During benchmark setup their runtime
format was changed uniformly from two to six decimal places so repeated-run
standard errors are not quantized to 10 milliseconds. Each task manifest
records both `source_evaluator_sha256` and the active `evaluator_sha256`, plus
`timing_precision_digits = 6`.

Git history first records the canonical task solutions in commit
`fb0f81bb00d7ad5c9e8ef33c66091eefecdb3f66`. It first records the two
reference-derived variants in commit
`b3b0c08f0bbc82470da5cbe39a6ba31e39f6a7fe`.

The source checkout has no configured Git remote. The published repository is
<https://github.com/sxzgroup/ORBIT-Q>. The performance task is
<https://github.com/QuantumBFS/quantum.harness/issues/78>.

Each benchmark `tasks/task-XX/task.toml` records:

- the source path in the parent ORBIT-Q repository;
- the SHA-256 digest of each solution;
- the copied evaluator's SHA-256 digest;
- the environment profile used to execute the task.

The public TensorCircuit-NG image does not expose the historical
`tc.set_contractor("omeco")` alias used by expert solutions 01, 10, and 12.
`envs/tensorcircuit-py311/sitecustomize.py` backports the official shortcut from
`tensorcircuit/cons.py` at TensorCircuit-NG commit
`53a712b517cdcaba69ca6376d9d68cd140bdeaea`: OMECo TreeSA with 16 trials, 32
iterations, the published score weights, and preprocessing enabled. The runner
records the shim hash as environment provenance. Reference and optimized
processes receive the same shim; no expert source is rewritten.

Run `./bench verify` after changing, importing, or regenerating task files.
The verifier rejects missing files, digest mismatches, invalid task IDs, and
solutions without `run_solution(config)`.

The parent Harbor task copies remain in place because Harbor treats each task as
its Docker context and `solution/solve.sh` expects the reference file there.
Removing those compatibility copies would break existing oracle runs.

The 2026-07-27 bootstrap deliberately preserves symmetric reference/candidate
failures for Tasks 06 and 08 rather than changing expert code or validity
rules to force a timing. See `baselines/bootstrap-2026-07-27.md`; these failures
close performance promotion for those tasks but do not invalidate the runner
parity check.
