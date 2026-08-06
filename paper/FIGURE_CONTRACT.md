# Figure contract

The paper update follows the visual and metric contracts of
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105).

1. Updated Fig. 1c preserves the agent-by-framework matrix, places all
   configurations in one continuous layout without an old/new separator, and
   includes the six added TensorCircuit-NG campaigns.
2. Updated Fig. 2b preserves failure rate and geometric-mean artifact runtime
   relative to the original public expert references, evaluated on passed tasks.
3. Updated Fig. 4 preserves the paper's 2 × 3 wall-time, token, and efficiency
   layout. All token bars use the same total-token encoding because the source
   tables do not publish components for every configuration.
4. The expert-optimization overview reports paired end-to-end measurements;
   exact task reductions are distinguished visually; Task 08 remains a valid
   optimized implementation with a small measured point estimate.
5. Fable 5 and DeepSeek max are excluded. Task 08 is failed for all six added
   solver configurations after final human review.
6. Figure labels omit the repeated `high` qualifier. The accompanying text
   states that Sol, Terra, Luna, DeepSeek V4 Flash, and Grok 4.5 use high
   thinking effort; the separately named Sol ultra configuration uses ultra.

Vector PDF is the publication deliverable; PNG copies are non-publication
previews for inline GitHub/PR display only.
Agent wall time, token use, artifact-runtime ratio, and expert-implementation
speedup remain separate quantities.
