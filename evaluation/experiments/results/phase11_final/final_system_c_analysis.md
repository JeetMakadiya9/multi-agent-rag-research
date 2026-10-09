# System C execution-path analysis

Successful C traces audit a bounded autonomous execution path with researcher gate. This is not fully autonomous research.

| Stage | PASS / audited successful traces |
|---|---:|
| phase_8a_validation | 276/276 |
| phase_8b_execution | 276/276 |
| phase_7b_dispatch | 276/276 |
| phase_6_retrieval | 276/276 |
| phase_6_verification | 276/276 |
| continuation_authorization | 276/276 |
| researcher_transition | 276/276 |
| completion | 276/276 |

C had 276 successful, 22 failed and 2 interrupted attempts. Attempt fields total 596 controller cycles over 298 records with measurements; interrupted values are unknown.


Trace discrepancy: for 22 failed C attempts, the attempt-level controller_cycles field is 2 while the embedded failure trace reports cycles_completed=1. Both values are preserved in final_system_c_analysis.json; this analysis does not reconcile them.
