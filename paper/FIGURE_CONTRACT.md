# Figure contract

The paper update follows the visual and metric contracts of
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105).

1. Updated Fig. 1c preserves the agent-by-framework matrix and appends only the
   five requested TensorCircuit-NG campaigns.
2. Updated Fig. 2b preserves failure rate and geometric-mean artifact runtime
   relative to the original public expert references, evaluated on passed tasks.
3. Updated Fig. 4 preserves the paper's 2 × 3 wall-time, token, and efficiency
   layout. All token bars use the same total-token encoding because the source
   tables do not publish components for every configuration.
4. The expert-optimization overview reports paired end-to-end measurements;
   exact task reductions are distinguished visually; Task 08 remains a valid
   optimized implementation with a small measured point estimate.
5. Fable 5 and DeepSeek max are excluded. Task 08 is failed for all five new
   solver configurations after final human review.

Vector PDF is the publication deliverable; PNG copies are non-publication
previews for inline GitHub/PR display only.
Agent wall time, token use, artifact-runtime ratio, and expert-implementation
speedup remain separate quantities.
