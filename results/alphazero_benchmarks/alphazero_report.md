# AlphaZero Benchmark Summary

Final model evaluated: `models_alphazero_two_phase/alphazero_final.pt`

## Best checkpoint (progression suite)

- Best vs Random: iter 50 (100.0%)
- Best vs Strategic-0.5: iter 60 (36.7%)
- Best vs Minimax-0.5: iter 80 (20.0%)

## Final benchmark table

| Opponent | Games | Win rate |
|---|---:|---:|
| Random | 100 | 99.0% |
| S03 | 100 | 48.0% |
| S05 | 100 | 27.0% |
| S07 | 100 | 14.0% |
| MM03 | 100 | 55.0% |
| MM05 | 100 | 15.0% |
| MM10 | 100 | 0.0% |

## Output files

- `alphazero_progression.png`
- `alphazero_final_results.png`
- `alphazero_checkpoint_results.csv`
- `alphazero_final_results.csv`
