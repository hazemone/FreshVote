# FreshVote primary-evaluation validation

**Status:** PASS

- Data rows: 3,840 / 3,840
- Unique policy-condition-seed keys: 3,840 / 3,840
- Factor cells with 40 held-out seeds: 96 / 96
- Frozen manifest exact match: True
- Log entries reconciled to CSV: True
- Full completion marker present: True

## Outcome counts

| Policy | Stable correct | Stable incorrect | No stable outcome |
|---|---:|---:|---:|
| baseline | 960 | 0 | 0 |
| hard_ttl | 960 | 0 | 0 |
| fixed_decay | 960 | 0 | 0 |
| adaptive_freshvote | 960 | 0 | 0 |

## Integrity hashes

- `manifest`: `3e4356587e9d1d56dff311d366c4e9589038f00fa3286c36accf0033c4d67012`
- `evaluation_csv`: `9950b5d9dc363d073a93a6ae1af5f66d6b1892a458fd0e321a5ee40743038362`
- `execution_log`: `9a4740d666ad065ebb278119851960236329fb85809e4a10b7849d9a9733636b`

## Reproducibility warning

The archive does not contain Python/NumPy versions or pip freeze; request them from the execution machine for the reproducibility record.

The raw uploaded archive was not modified. This validation uses only the held-out primary-evaluation files.
