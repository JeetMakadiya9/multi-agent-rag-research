# Efficiency analysis

Latency separates SUCCESS and FAILED attempts. INTERRUPTED rows have no recorded latency and are not imputed. Call totals are sums of persisted non-null fields.

| System | Success latency mean / median / SD / range (s) | Failed latency mean / median / range (s) | LLM | Retrieval | Verification | Controller cycles |
|---|---|---|---:|---:|---:|---:|
| A | 26.49 / 26.15 / 3.64 / 16.95–37.54 | 9.58 / 8.90 / 5.60–19.44 | 299 | 299 | 0 | N/A |
| B | 39.64 / 38.25 / 9.71 / 24.31–110.98 | 27.10 / 16.44 / 11.76–73.21 | 645 | 298 | 298 | N/A |
| C | 38.58 / 36.96 / 8.83 / 22.51–89.20 | 26.03 / 21.73 / 11.39–50.76 | 608 | 596 | 298 | 596 |
