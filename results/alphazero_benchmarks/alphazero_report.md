# AlphaZero Benchmark Summary

Final model evaluated: `models_alphazero_two_phase/alphazero_final.pt`

## Best checkpoint (progression suite)

- Best vs Random: iter 30 (100.0%)
- Best vs Strategic-0.5: iter 70 (46.7%)
- Best vs Minimax-0.5: iter 100 (30.0%)

## Final benchmark table

| Opponent | Games | Win rate |
|---|---:|---:|
| Random | 100 | 98.0% |
| DQN140 | 100 | 42.0% |
| S03 | 100 | 70.0% |
| S05 | 100 | 26.0% |
| S07 | 100 | 14.0% |
| MM03 | 100 | 61.0% |
| MM05 | 100 | 21.0% |
| MM10 | 100 | 0.0% |

## Output files

- `alphazero_progression.png`
- `alphazero_final_results.png`
- `alphazero_checkpoint_results.csv`
- `alphazero_final_results.csv`
