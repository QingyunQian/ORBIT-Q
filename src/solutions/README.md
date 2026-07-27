# Optimized expert solutions

Each `challenge-XX/solution_N.py` starts as a byte-for-byte copy of the frozen
human expert reference. Autoresearch agents edit one of these files in a
task-specific Git worktree.

Measure the candidate and compare it with the frozen reference:

```bash
./bench run 01 \
  --solution optimized \
  --compare-to reference \
  --repeat 6
```

The comparison qualifies only when every measured reference and candidate run
passes the original evaluator.
