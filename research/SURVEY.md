# ORBIT-Q Runtime Optimization Survey

**Status: NOT READY**

The research gate remains closed. A maintainer must replace each `TODO` below
with cited evidence before an agent edits an optimized solution or runs a
candidate experiment.

## Evidence rules

Use one of these evidence forms for each factual claim:

- External source: title, author or project, version or commit, section, and URL.
- Repository source: commit, path, symbol or line, and file SHA-256.
- Local measurement: report path, manifest SHA-256, cell IDs, host fingerprint,
  image digest, and command.

Mark a source statement as `source claim` and a reproduced observation as
`local evidence`. Record `evidence gap` when no source or reproduction supports
a comparison. Do not describe a method as state of the art without a named,
reproduced comparator under matched conditions.

Primary scope:

- [Quantum Harness issue #78](https://github.com/QuantumBFS/quantum.harness/issues/78)
- The task contracts and evaluators under `../tasks/challenge-XX/`
- The human expert sources under `../references/challenge-XX/`

## Cited-evidence checklist

- [ ] All 12 task sections identify the expert algorithm and output contract.
- [ ] All 12 sections cite the evaluator and define its timed region.
- [ ] All complexity and scaling claims cite a paper, source implementation, or
      measured profile.
- [ ] Framework claims cite inspected TensorCircuit-NG, JAX, JAXLIB, OMECo,
      TensorNetwork-NG, Quimb, or contractor source at a pinned version.
- [ ] Each bottleneck claim includes profile evidence or carries an
      `evidence gap` label.
- [ ] Each optimization hypothesis states the quantum semantics and output
      fields it must preserve.
- [ ] The environment inventory records package versions, source commits, image
      digest, Dockerfile hash, and requirements-lock hash.
- [ ] The measurement section freezes paired run order, repeat count, timeout,
      confidence method, and promotion rule before candidate results exist.
- [ ] The cross-task section separates compile cost, contraction cost, optimizer
      work, sampling, and result assembly.
- [ ] The survey names each external baseline required for any SOTA, 10x, 100x,
      or scaling claim.

## Environment and framework inventory

| Component | Version or commit | Inspected paths and symbols | Evidence |
| --- | --- | --- | --- |
| TensorCircuit-NG | TODO | TODO | TODO |
| JAX | TODO | TODO | TODO |
| JAXLIB | TODO | TODO | TODO |
| OMECo | TODO | TODO | TODO |
| TensorNetwork-NG | TODO | TODO | TODO |
| Quimb | TODO | TODO | TODO |
| Benchmark image | TODO | TODO | TODO |

## Challenge 01: DMRG-MPS Input With Variational Circuit Refinement

- Expert and contract: TODO, cite `../references/challenge-01/solution_1.py`,
  `../tasks/challenge-01/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 02: Entanglement-Profile-Constrained VQE

- Expert and contract: TODO, cite `../references/challenge-02/solution_2.py`,
  `../tasks/challenge-02/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 03: Probability-Aware Post-Selected Many-Body Cooling

- Expert and contract: TODO, cite `../references/challenge-03/solution_3.py`,
  `../tasks/challenge-03/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 04: Trainable Kraus Noise Calibration From Multi-Circuit Data

- Expert and contract: TODO, cite `../references/challenge-04/solution_4.py`,
  `../tasks/challenge-04/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 05: Custom Non-Unitary Gate Cooling

- Expert and contract: TODO, cite `../references/challenge-05/solution_5.py`,
  `../tasks/challenge-05/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path, OMECo path, and compile behavior: TODO with cited source
  symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 06: Digital-Analog Hybrid VQE With Trainable Analog Blocks

- Expert and contract: TODO, cite `../references/challenge-06/solution_6.py`,
  `../tasks/challenge-06/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 07: 16-Qubit Measurement-Feedback VQE

- Expert and contract: TODO, cite `../references/challenge-07/solution_7.py`,
  `../tasks/challenge-07/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 08: 7x7 Mixed-Axis Grid Tensor-Network Sampling

- Expert and contract: TODO, cite `../references/challenge-08/solution_8.py`,
  `../tasks/challenge-08/problem.md`, and the evaluator.
- Algorithm, contraction scaling, and memory: TODO with cited evidence.
- Timed contractor path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 09: Random Local Light-Cone Optimization

- Expert and contract: TODO, cite `../references/challenge-09/solution_9.py`,
  `../tasks/challenge-09/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 10: 22-Qubit VQE With an 18-Qubit Controlled-Z Hyperedge

- Expert and contract: TODO, cite `../references/challenge-10/solution_10.py`,
  `../tasks/challenge-10/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 11: Spin-1 Haldane-Chain VQE With String-Order Verification

- Expert and contract: TODO, cite `../references/challenge-11/solution_11.py`,
  `../tasks/challenge-11/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework path and compile behavior: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Challenge 12: Variational Circuit to MPS Overlap Optimization

- Expert and contract: TODO, cite `../references/challenge-12/solution_12.py`,
  `../tasks/challenge-12/problem.md`, and the evaluator.
- Algorithm, complexity, and memory: TODO with cited evidence.
- Timed framework and MPS contraction paths: TODO with cited source symbols.
- Profiled bottleneck, hypotheses, and preserved semantics: TODO.

## Cross-task optimization evidence

TODO: compare reusable batching, vectorization, `jit` and `vmap` placement,
scan structure, contractor selection, tensor layout, sparse operations, device
placement, compilation cache behavior, and memory limits. Cite each comparison.

## Frozen measurement and statistics plan

TODO: cite and freeze the paired-run method, run order, workload
version, repeat count, validity rule, confidence interval, and promotion rule.
The survey cannot change this section after candidate timing results exist.

## Claims boundary

TODO: list reproduced external comparators and their matched provenance. Until
that list exists, report gains only against the bundled human expert reference.
