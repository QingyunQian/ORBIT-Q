# Task 01 Autoresearch Campaign

Task: `task-01`

Campaign branch: `codex/orbitbreakers/task-01/extreme-native`

Reference parent: `0819ed34dd3f1eaa8f77587c2a40d95a420ea829`

## Objective

Optimize the immutable Task 01 human expert to the limit while preserving the
full DMRG-MPS input, 570-parameter circuit, 500 Adam updates, history semantics,
TensorCircuit-NG framework fidelity, and exact original docstring. The older
MPO candidate on `main` is a historical factor, not the campaign parent.

## Initial gate status

At `2026-07-29T14:59:00Z`,
`python3 research/check_gates.py --task 01 --json` reported:

- Task 01 `SURVEY.md`: missing;
- public manifest selection/coverage for Task 01: missing;
- current repeated immutable-reference report: missing;
- `research_ready`: false.

No candidate source was changed while these gates were closed.

## Public overlap check

Open upstream PRs inspected at `2026-07-29T14:59:00Z`: #3, #4, #5, #6, and
#7. No active PR targets Task 01 expert-runtime optimization.

## Append-only campaign events

- `2026-07-29T14:59:00Z`: created the Task 01 campaign branch from current
  `origin/main`; recorded immutable source hashes, environment versions, and
  the gate failure; began a six-run immutable-reference baseline.
- `2026-07-29T15:04:00Z`: froze the public survey and predeclared twelve
  single-factor hypotheses.
- `2026-07-29T15:04:30Z`: the canonical public workload and six-run
  immutable-reference validation completed. All six runs passed; runtimes
  were `54.896941`, `56.031343`, `52.621569`, `55.264405`, `58.445563`,
  and `56.618919` seconds (mean `55.6464567 s`, median `55.647874 s`).
  Report SHA-256:
  `61aca162f2fd80634f253464e885d83ca16bbb7ffefaa99ade2e83b23412e8c2`.
  The Task 01 public dataset is version
  `orbitq-workloads-v20260729.8`; the survey is now `READY`.
- `2026-07-29T15:09:00Z`: immutable-reference profiler completed. Report:
  `research/task-01/profiles/reference-profile.json`, SHA-256
  `d21cac5b49052cbaa3a4bb286bb968682cc5158f7d0e62a8d78ffeda6141f417`.
  Lowering and compilation took `6.7701/22.6276 s`; the five-step steady mean
  was `0.0435980 s`, projecting to `21.7990 s` for 500 updates. XLA estimated
  `875,047,872` FLOPs and `624,489,344` bytes accessed per step. Both graph
  compilation and repeated gradient contraction are first-order costs.
- `2026-07-29T15:22:00Z`: screened an exact untruncated local-MPS candidate.
  Its non-JIT forward calculation produced an initial energy near
  `-41.504047`, consistent with the expert at complex64 precision, and its
  exact bond dimensions reached at most 128 as predicted. However, the
  candidate-first gradient/update equivalence job did not compile within a
  280-second hard window. Decision: discard the fully expanded MPS graph on
  end-to-end compilation grounds; retain the algebraic result as negative
  evidence.
- `2026-07-29T15:33:00Z`: experiment `e01`, direct bond-three TFIM MPO as the
  sole change from the immutable expert, completed five alternating pairs.
  All ten runs passed. Reference mean/median:
  `55.406736/53.550141 s`; candidate mean/median:
  `36.6112458/36.625840 s`. Every pair favored the MPO. Mean paired speedup
  `1.512425x` (standard error `0.049414x`) and mean paired reduction
  `33.6176%`. Report SHA-256:
  `bab25b3dc7545eb713b627071a30954555ab729966363d1fdef57132948fe8f7`.
  Decision: retain as a major factor, pending final six-pair promotion.
- `2026-07-29T15:41:00Z`: experiment `e02` paired TensorCircuit
  `K.jaxy_scan` against the otherwise identical retained MPO parent for five
  alternating pairs. All runs passed. Parent mean/median:
  `36.6960228/36.474965 s`; scan mean/median:
  `38.1729492/36.113780 s`. Mean paired speedup was only `0.966926x`,
  with 2/5 wins and one 45.26-second tail. Report SHA-256:
  `61180e1fe14b3d934630de87f484f49340c8290ef0ae891b610e316e564db910`.
  Decision: discard; it does not establish a runtime gain.
- `2026-07-29T15:48:00Z`: experiment `e03` explicitly fused each
  `RZ-RY-RZ` triple and each commuting `RXX-RYY-RZZ` triple, then combined the
  site gates with their disjoint layer entangler. All five alternating pairs
  against the retained MPO parent passed and favored fusion. Parent
  mean/median: `39.0692418/39.093710 s`; fused mean/median:
  `24.9974732/25.171011 s`. Mean paired speedup `1.562085x`
  (standard error `0.016403x`) and reduction `35.9544%`. Report SHA-256:
  `9c54954cbaa98743cd76db2ce2112573e7fd65a600fcdf58b82f2c5bb816a72a`.
  Decision: retain and promote into the campaign candidate.
- `2026-07-29T15:55:00Z`: experiment `e04` reduced OMECo TreeSA from the
  compatibility default `16 trials x 32 iterations` to `1 x 1`, keeping
  preprocessing and all contraction operands unchanged. The preliminary pair
  improved by 13.43%; all five formal pairs subsequently passed and favored
  `1x1`. Default mean/median: `26.5024358/24.329149 s`; `1x1` mean/median:
  `23.8723832/23.151031 s`. Mean paired speedup `1.103946x` and reduction
  `9.1091%`. Report SHA-256:
  `e38b4fa0adb9d89830d8abd48768238e9b9e4d6052fceec8c626d59fa620445d`.
  Decision: retain, then screen larger search budgets and greedy.
- `2026-07-29T16:02:00Z`: experiment `e05` compared OMECo `4x4` with retained
  `1x1` for five pairs. Both means were noisy and close:
  `25.1811468 s` for `4x4` versus `25.5150162 s` for `1x1`; mean paired
  speedup `1.015110x`, only 3/5 wins, standard error `0.032096x`, and two
  pairs lost. Report SHA-256:
  `47cfe0e85048573a15db24e3fc5e629e3c08d32bb457b696ea0755f28100caa3`.
  Decision: discard because the gain is not established and fails the 5/6-like
  directional criterion.
- `2026-07-29T16:03:00Z`: experiment `e06` greedy+preprocessing quick screen
  passed physically but took `25.627988 s` versus `21.819974 s` for OMECo
  `1x1`, a `0.851412x` paired speedup. Report SHA-256:
  `a45ce9fe39f346d827d2834ac9221618d03ac5505553df99a6550f01f4a788a7`.
  Decision: discard without a five-pair campaign because it is 17.45% slower
  in the predeclared quick screen.
- `2026-07-29T16:10:00Z`: experiment `e07` batched all one-qubit Euler
  matrices per layer and all commuting two-qubit interaction matrices per
  layer, using exact closed forms and TensorCircuit backend operations. Two
  test angle rows matched native TensorCircuit gates to `6.14391e-8` for the
  one-qubit matrices and exactly for the two-qubit matrices. On one shared DMRG
  input, the complete 500-step histories differed by at most
  `1.56403e-4` (`3.76836e-6` relative); both initial and final values remained
  in the official window. Equivalence record SHA-256:
  `88978e1078261b5ecba7cdcbdb6fd5c64446470c2707d36b218afd65f47060ec`.
- `2026-07-29T16:13:00Z`: the `e07` five-pair runtime ablation passed all ten
  runs and favored batching in 5/5 pairs. Native-fused parent mean/median:
  `21.9768618/21.751241 s`; batched candidate mean/median:
  `6.153842/6.206690 s`. Mean paired speedup `3.575483x` (standard error
  `0.060419x`) and reduction `71.9996%`. Report SHA-256:
  `67ec0d61548af115ca5f69a32d90edfa78ceede8f94f9bc94cc96e4952602ebc`.
  Decision: retain as the dominant factor and promote into the campaign
  candidate.
- `2026-07-29T16:17:00Z`: experiment `e08` additionally batched the 62
  Kronecker products and 4x4 matrix products within their layers. Five-pair
  means were `5.7539312 s` versus `5.9601554 s`; mean paired speedup
  `1.037608x`, but only 4/5 pairs won and the standard error was
  `0.023024x`. Report SHA-256:
  `82ca883f04e8b3ab92568df63f933a9b02e16448d892bdb192b1add99945beab`.
  Decision: do not retain this isolated form because its gain is not
  established; test a broader cross-layer batch instead.
- `2026-07-29T16:20:00Z`: experiment `e09` retested whole-training
  `K.jaxy_scan` after graph compression. Five-pair means were
  `5.8107638 s` for scan and `5.9035310 s` for the Python-dispatched compiled
  step. Mean paired speedup `1.017324x`, only 3/5 wins, and two losses.
  Report SHA-256:
  `f8d18c294403f10019e66887c9a8f6b168142ffdd905b17448b2002df1b29f13`.
  Decision: discard again; graph compression changes the magnitude but not the
  evidentiary conclusion.
- `2026-07-29T16:22:00Z`: the first `e10` cross-layer batching screen failed
  before timing because a `[4,32,3]` parameter tensor was passed to a helper
  expecting `[batch,3]`; report SHA-256:
  `af835fa34a4bf0dd09699c43708c0d33e24acd195f8bb1dc6470ee0c4167d3ef`.
  The helper input was flattened to `[128,3]` and reshaped after gate assembly.
  The failed run is preserved and excluded from performance claims.
- `2026-07-29T16:28:00Z`: corrected experiment `e10` assembled all 128
  one-qubit matrices and all 62 two-qubit matrices in two global backend
  batches, then batched the disjoint pair products per layer. In six
  alternating pairs against the retained per-layer-batched parent, all runs
  passed and all six pairs favored global batching. Parent mean/median:
  `5.8173395/5.6605435 s`; candidate mean/median:
  `5.4079000/5.4058405 s`. Mean paired speedup was `1.079381x`
  (standard error `0.028711x`, 95% Student-t interval
  `[1.005576, 1.153186]`). Decision: retain.
- `2026-07-29T16:34:00Z`: experiment `e11` enabled TensorCircuit-NG's
  algebraic contractor path with `use_primitives=True` and otherwise kept the
  retained candidate fixed. Its one-pair screen was positive, but the formal
  six-pair run won only 4/6 pairs. Parent/candidate means were
  `6.0225168/5.0355388 s`; one parent outlier at `8.638289 s` inflated the
  mean speedup to `1.204885x`, while two pairs lost and the 95% Student-t
  interval was `[0.867527, 1.542243]`. Decision: discard because the
  predeclared directional and uncertainty gates both failed.
- `2026-07-29T16:40:00Z`: final six alternating matched pairs compared the
  immutable expert with the promoted e10 candidate. All 12 evaluator cells
  passed and the candidate won 6/6 pairs. Expert mean/median:
  `60.6511438/62.4723120 s`; candidate mean/median:
  `6.3598067/6.5868035 s`. Mean paired speedup was `9.636410x`
  (standard error `0.371756x`, 95% Student-t interval
  `[8.680782, 10.592037]`) and mean paired time reduction was `89.5397%`.
  Raw report SHA-256:
  `6c3c31370fedc100e061327e52cbcdcfc6e3709b8ba2b41310e829669a7ab237`.
- `2026-07-29T16:44:00Z`: final candidate profile completed. Lowering and
  compilation fell to `0.5416/2.1644 s`; five steady updates averaged
  `0.0057078 s`; the StableHLO contained 2,754 lines. XLA estimated
  `132,386,304` FLOPs and `49,688,780` bytes accessed per update. Profile
  SHA-256:
  `eeca4de572f324f5063fb4c19dd3b8d376f042eca01e438ef8ef8129a278de34`.
- `2026-07-29T16:47:00Z`: the first consolidated final-equivalence harness
  reused one mutable TensorCircuit MPS/MPO object across two independent JAX
  traces. Tensor-network contraction left a traced tensor attached to the
  shared object, so the second trace correctly failed with
  `UnexpectedTracerError`. No numerical result was emitted. The harness was
  corrected to build independent native and candidate MPS/MPO objects from
  the same immutable DMRG state.
- `2026-07-29T16:49:00Z`: the corrected consolidated equivalence run
  completed all 500 updates. Closed-form one-/two-qubit gates differed from
  native TensorCircuit products by at most `6.143906e-8/0`; the full histories
  differed by at most `1.716614e-4` absolute and `4.136032e-6` relative.
  Both trajectories remained in the canonical evaluator window. Record
  SHA-256:
  `fa694c68c939f67ade456b9f7603045fc585d1975195d2a61df887cf1810ad8e`.

## Corrections

Append corrections here; never rewrite an experiment after it informs a later
choice.

- `2026-07-29T15:23:00Z`: the first attempted exact-MPS equivalence command
  was initially described as compiling the candidate, but traceback proved it
  was still compiling the reference objective. Interrupting the host command
  also left its temporary Docker container alive. That container was stopped
  before the candidate-first 280-second screen. No timing from the first
  attempt is used.
- `2026-07-29T16:41:00Z`: the `e02` narrative above says 2/5 scan pairs won.
  Re-reading the immutable report gives speedups `0.999992`, `0.991897`,
  `1.014756`, `0.979015`, and `0.848971`; strictly greater than one is
  therefore 1/5, not 2/5. The discard decision is unchanged.
