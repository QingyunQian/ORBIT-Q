# Task 02 Autoresearch Campaign

Task: `task-02`

Campaign branch: `codex/orbitbreakers/task-02/extreme-native`

Campaign task: `task-02`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign objective

Reduce evaluator-reported runtime for the immutable Task 02 human expert while
preserving all 243 trainable parameters, three block-checkpoint Renyi-2
entropies, the 45-term XXZ energy, and exactly 500 sequential Adam updates.

## Selection and gates

- `2026-07-29T13:30Z`: inspected all open PRs on `sxzgroup/ORBIT-Q` and the
  benchmark repository. No active Task 02 optimization was present. This
  campaign and every hypothesis worktree are bound to Task 02.
- Base commit:
  `d13e2591574cc1480507b00bcb33b0c6a48e6b99`.
- Public data only; no hidden or private evaluation is used.
- Reference and initial optimized sources were byte-identical at selection,
  SHA-256
  `cd5776dfb223924edd83795bd8222032351008a584b55d747110521c709dcdd3`.

## Baseline `reference-six-20260729`

Hypothesis: the latest TensorCircuit-NG image and fixed 6-CPU/7-GiB envelope
produce a valid repeated reference suitable for paired promotion decisions.

Command:

`./bench run 02 --solution reference --repeat 6 --engine docker --cpus 6
--memory 7g --timeout 300 --no-build --output
results/task-02-reference-20260729`

Environment:

- image ID:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`;
- TensorCircuit-NG `1.8.0.dev20260726`;
- JAX/JAXLIB `0.10.0` / `0.10.0`;
- Optax `0.2.8`;
- TensorNetwork `0.5.1`; Quimb `1.11.1`;
- host fingerprint:
  `c627504db97dc65b8d998afb4b9cdf73cfc3eff2ce2ac589b4a7a4aa0c7fdc48`;
- one shared container, six CPUs, 7 GiB, no network, 300-second per-cell cap;
- evaluator SHA-256:
  `0d9360d4812d033d364253dd478444f68918f1b05f04146eddcb846a3dca3a1e`;
- dependency-lock SHA-256:
  `cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`.

Result:

```text
terminal_status: SUCCESS
valid: true
timed_out: false
passing_runs: 6/6
runtimes_sec: 4.559075, 4.986945, 4.990731, 5.188701, 4.813289, 6.035628
mean_runtime_sec: 5.0957281667
median_runtime_sec: 4.988838
sample_stdev_sec: 0.5064942904
runtime_stderr_sec: 0.2067754282
min_runtime_sec: 4.559075
max_runtime_sec: 6.035628
```

Immutable report SHA-256:
`7d450e074cc5a4cabca72e9f6cd5787e2f5226c42c69cf6c3016ec53465b9912`.

Decision: `baseline`. The repeated-reference promotion gate passes. No
improvement is claimed without matched candidate pairs.

## Append-only campaign events

- `2026-07-29T13:31Z`: inspected installed TensorCircuit sources. The expert's
  Pauli MVP traces one path per Hamiltonian term; Renyi-2 forms `rho @ rho`;
  and the backend's `K.jaxy_scan` delegates to `jax.lax.scan`.
- `2026-07-29T13:31Z`: completed the cited survey and canonical public workload
  record. Candidate editing remains blocked until the manifest is rebound and
  the fail-closed gate passes.
- `2026-07-29T13:33Z`: survey, public dataset, and repeated-reference gates all
  pass. Added an immutable-reference profiler to separate lowering,
  compilation, trajectory, entropy, Hamiltonian action, and steady optimizer
  execution before selecting candidate factors.
- `2026-07-29T13:35Z`: immutable-reference profile completed. Report
  `research/task-02/profiles/reference-profile.json`, SHA-256
  `b7d190ea225ca5586d813759b89fea0cba05c045ca41e72e8f687a8769241d40`;
  profiler SHA-256
  `538a86104c32925586606394f50be069800c18b054bae45a6ebb786ac0e76a8b`.
  One update lowers in `0.803158 s`, compiles in `1.929769 s`, contains
  15,479 StableHLO lines, and executes in `0.002643 s` mean over 12 steady
  calls. The projected 500-update kernel time is `1.321653 s`. The forward
  trajectory accounts for about 93.5% of the separately measured loss
  execution (`0.000889/0.000951 s`); a Hamiltonian action is only
  `0.000176 s`, and one generic entropy is `0.000132 s`. Decision: prioritize
  graph reduction and host-loop removal, while treating sparse Hamiltonian and
  purity rewrites as secondary measured factors.

## Append-only corrections

Append corrections below this heading. Never rewrite an earlier result after
it has informed another experiment.

## Experiment `e07-batched-purity`

Branch: `codex/orbitbreakers/task-02/e07-batched-purity`

Parent commit: `edf1e91` (accepted whole-training scan).

Execution note: this hypothesis uses a fresh local shared Git checkout and
independent branch at the accepted commit after worktree permission review
timed out.

Hypothesis: the predeclared shared-kernel combination of checkpoint
`K.vmap` and exact Frobenius purity will batch the three reduced-density
matrices while removing all three dense `rho @ rho` products. The prior
single-factor reports remain the subfactor ablations; neither subfactor alone
passed the confidence rule.

Permitted data: public Task 02 artifacts and the two completed single-factor
reports only. No hidden/private evaluation.

Pre-evaluation correctness rule: initial loss/auxiliary values, gradient, and
all four 12-step histories must satisfy the frozen survey tolerances.

Candidate source SHA-256:
`181c0459c8194aa259be813e23896d3c70691adf5999a26353fe57cab14d3f3d`.

Candidate source-diff SHA-256:
`ad7f082c18273b9f8c67cab5aef8477983791011261bb604ea62f6768d3a3fcf`.

The 12-step audit passed: initial loss error `0`, auxiliary maximum error
`2.981e-7`, gradient maximum error `1.211e-8`, and maximum history error
`1.073e-6`. Audit report SHA-256:
`107c59ce1785ee9ada91e061d01970f1522ea73f3d8fe3efea37c03138d06b53`.

Six direct parent/factor pairs all passed. Parent/factor means were
`4.531128/4.218862 s`; factor wins `6/6`; mean paired speedup `1.074457x`,
95% Student-t interval `[1.044174x, 1.104740x]`. The runner roles were
intentionally reversed so its registered `optimized` source was the factor;
the tracked report normalizes every ratio as parent/factor. Raw report
SHA-256:
`6dd10f3c24054668734022000058391c192c87f0554d404a6d3b7d61593ee07e`.

Decision: `keep`. The shared-kernel combination passes the confidence rule
although each predeclared subfactor was inconclusive alone.

## Experiment `e02-training-scan`

Branch: `codex/orbitbreakers/task-02/e02-training-scan`

Parent commit: `77939b1`

Hypothesis: carrying parameters and Optax state through one TensorCircuit
`K.jaxy_scan`, compiled by `K.jit`, will preserve all 500 sequential updates
and pre-update histories while removing 500 Python-to-device dispatches.

Permitted data: public Task 02 artifacts and tracked reference profile only.
No hidden/private evaluation.

Candidate SHA-256:
`6e44df512170e071655eee7697f7a0c084704dd34c266f441b2755cb1f29bc1a`.

Candidate diff SHA-256:
`9b590f4cb193a39327b38165bcd255f3e051f5eb76d495b141194ebefbcbe1d8`.

Pre-evaluation correctness rule: all four 12-step pre-update histories must
match the immutable expert within `2e-4` and retain identical shapes.

The 12-step audit passed with bit-identical values and shapes for all four
returned histories. Audit report SHA-256:
`8864dd25031e00e1df188f705824738925e3679e4e46e06ae7078c5ccb6a1493`.

Six alternating canonical pairs all passed. Reference/candidate means were
`4.724217/4.563051 s`; candidate wins `6/6`; mean paired speedup
`1.035580x`, 95% Student-t interval `[1.010063x, 1.061098x]`. Raw report
SHA-256:
`549e267378bc1eca2ea2b839810af4c46780ac8bdc7800eacdfc15385bc1e33b`.

Decision: `keep`. Whole-training `K.jaxy_scan` is the first accepted parent
and removes about 3.41% of end-to-end runtime in this session.

## Consolidated independent-factor records

The following experiments ran on isolated branches. They are appended here
after completion so the final branch retains both successful and negative
evidence without promoting discarded code.

### Experiment `e01-gate-fusion`

Parent commit: `77939b1`. Candidate SHA-256:
`76917fc2292984b0102dda14ceb3b873f4ca55779554d29f2fa887ffd2f3f214`.

Hypothesis: replace all exact `RY -> RZ` pairs and commuting
`RXX -> RYY -> RZZ` triples with one differentiable TensorCircuit unitary,
reducing circuit applications from 243 to 105 without tying parameters.

The final audit passed: local matrix error `1.33e-7`, state error `3.07e-7`,
loss error `2.38e-7`, gradient error `7.45e-8`, and post-update physical
output error `5.36e-7`. The raw Adam parameter difference (`3.63e-4`) came
from near-zero-gradient sensitivity and did not alter observables. Audit
report SHA-256:
`0c389fba34386ab57904a833d90f1192fe174515bfc5c928e27f44fb1e670a33`.

Six pairs: reference/candidate means `4.659226/4.825325 s`, `0/6` wins,
mean paired speedup `0.965849x`, 95% interval
`[0.946219x, 0.985478x]`. Raw report SHA-256:
`1804ebe44515690e1985411b16a51153aa6ddb88ec8a6c964d4d90fe0c365ce9`.

Decision: `discard`. Dynamic trigonometric and matrix assembly outweighed the
smaller gate-application count.

### Experiment `e03-purity`

Parent: accepted scan. Candidate SHA-256:
`ab560235d23e2dcd5256467f1dba03bb4eee7d675250ccf9054c3d8a670c508e`.

Hypothesis: use the exact Hermitian identity
`trace(rho @ rho) = sum(rho * conj(rho))`, removing three 64-by-64 matrix
products per loss.

The audit passed with gradient error `1.70e-8` and maximum 12-step history
error `1.25e-6`. Six pairs: parent/factor means
`4.591056/4.491962 s`, `5/6` wins, mean paired speedup `1.024675x`, 95%
interval `[0.961881x, 1.087470x]`. Raw report SHA-256:
`cccabf99871f8fe7d62ac42fdedd6d77394157fdd41a490ff42f324104565184`.

Decision: `discard_inconclusive`; the interval crosses one.

### Experiment `e04-sparse-xxz`

Parent: accepted scan. Candidate SHA-256:
`c5997e6147212271ec45321880a14d696849a6436fb99f16a02b72fc42992942`.

Hypothesis: replace 45 termwise MVP traces with TensorCircuit
`PauliStringSum2COO(..., numpy=True)` plus `K.sparse_dense_matmul`.

The audit passed with Hamiltonian-action error `1.67e-5`, gradient error
`3.91e-8`, and maximum 12-step history error `1.19e-6`. Six pairs:
parent/factor means `4.296697/4.417261 s`, `2/6` wins, mean paired speedup
`0.972882x`, 95% interval `[0.947782x, 0.997981x]`. Raw report SHA-256:
`8bfba5d0d90cde8ddb571e38d975044e5ea50001e9581a1a617a13fe9bca6c12`.

Decision: `discard`. For this 12-qubit, 45-term workload the native sparse
construction/kernel is slower than the current termwise MVP.

### Experiment `e05-packed-params`

Parent: accepted scan. Candidate SHA-256:
`895ede0c511e60221400a85066b036ab57fce683df614a2cabb389e928826b70`.

Hypothesis: pack ten parameter leaves into one `(3, 81)` tensor to shrink scan
carry, Optax state, PyTree bookkeeping, and the compiled graph.

The audit was bit-identical for initialization, loss, auxiliary values,
gradient, and four 12-step histories. Six pairs: parent/factor means
`4.592900/10.730835 s`, `0/6` wins, mean paired speedup `0.428014x`, 95%
interval `[0.411501x, 0.444527x]`. Raw report SHA-256:
`b24d2fbeb2094e23f2353b94a71896dc6b1f506bf6eea600b14536aa7d7617fb`.

Decision: `discard`. Static unpack slices add costly gather/scatter autodiff;
the original ten-leaf PyTree is substantially better.

### Experiment `e06-entropy-vmap`

Parent: accepted scan. Candidate SHA-256:
`c9c7840740152f03f23ddbc50379098847508007288da397ecf592952204cdb2`.

Hypothesis: return all three checkpoint states, then apply unchanged
TensorCircuit reduced-density-matrix Renyi-2 kernels through `K.vmap`.

The audit passed with gradient error `1.86e-8` and maximum 12-step history
error `8.34e-7`. Six pairs: parent/factor means
`4.765934/4.572339 s`, `5/6` wins, mean paired speedup `1.046783x`, 95%
interval `[0.985861x, 1.107705x]`. Raw report SHA-256:
`a43cd8242d76503057c88e4de2ee7b702965ecf89c41643b21ba4d8664d13a97`.

Decision: `discard_inconclusive`. It is not retained alone; E07 separately
predeclared and measured the changed shared kernel formed by combining this
factor with exact Frobenius purity.

## Final TensorCircuit-native normalization

After factor selection, the inner block scan was changed from direct
`jax.lax.scan` to TensorCircuit backend `K.jaxy_scan`, and the now-unused
direct JAX import was removed. In the installed backend this is a thin
framework-native delegation to the same JAX primitive, so it is a fidelity
normalization rather than a separately claimed speedup factor. The final
source SHA-256 before confirmation is:
`aef3652f8d80ec6f3e414f9496295b8b852db2b256ec414f4146b3a636485b30`.

The full E07 equivalence audit and final immutable-reference timing are rerun
after this normalization; no earlier factor result is reused as final timing.

## Experiment `e08-purestate-purity`

Parent commit: `0501d90`. Candidate source SHA-256:
`950e86b9c0db25497034ce0ff385619262bbdecfee56e35bcf832f13f95da7e3`.

Hypothesis: exploit the contiguous six/six public half-chain cut by reshaping
all checkpoint pure states to `(3, 64, 64)`, constructing retained-half Gram
matrices in one backend `K.einsum`, and computing their purities together.

The audit failed the frozen gate before timing. Initial loss error was
`5.96e-7` and gradient error `3.77e-8`, but the changed complex64 contraction
order produced auxiliary error `2.50e-6` and 12-step entropy-history error
`5.35e-6`, above the predeclared tolerance. Audit report SHA-256:
`d032009a567fcf2ec1da244318e27a8f1788a20c19af029b6afb8cc0eaa08b4d`.

Decision: `discard_before_timing`. No performance claim or chart is produced
for a candidate that failed correctness.

## Final immutable-reference confirmation

Final candidate SHA-256:
`aef3652f8d80ec6f3e414f9496295b8b852db2b256ec414f4146b3a636485b30`.

The post-normalization 12-step audit reproduced the E07 values: loss error
`0`, auxiliary error `2.981e-7`, gradient error `1.211e-8`, and maximum
history error `1.073e-6`.

All twelve cells in six alternating immutable-reference/candidate pairs
passed. Reference/candidate means were `4.495463/4.031039 s`; candidate wins
`6/6`; mean paired speedup `1.115649x`, 95% Student-t interval
`[1.057686x, 1.173613x]`; mean paired improvement `10.192992%`. The first
five pairs independently give mean runtimes `4.506090/4.039244 s` and mean
paired speedup `1.116107x`. Raw report SHA-256:
`27ec959f8e23156a0c8f0b15bb0f122b86e3a32e1842764888ef90eb07a842dd`.

Decision: `final_keep`.
