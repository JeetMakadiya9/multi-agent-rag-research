# Failure Analysis

Raw counts: {'SUCCESS': 840, 'INTERRUPTED': 5, 'FAILED': 55}. Non-success attempts are 60 and are not classified as wrong predictions.

By system: `{"A": {"interruption": 1, "local_model_endpoint_failure": 15}, "B": {"interruption": 2, "local_model_endpoint_failure": 14, "verification_failure": 4}, "C": {"controller_failure": 22, "interruption": 2}}`

By category: `{"controller_failure": 22, "interruption": 5, "local_model_endpoint_failure": 29, "verification_failure": 4}`

See `failure_analysis.json` and `failure_records.csv` for exact persisted errors. Classification accuracy excludes failed/interrupted and invalid/missing predictions, while reporting their counts. Retrieval unavailable outputs are not retrieval misses. No retries or inferred failure stages were added.
