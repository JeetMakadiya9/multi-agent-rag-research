# Multi-Agent RAG Research Studio

**An evidence-grounded research and claim-verification framework powered by multi-agent orchestration.**

[Open the Research Studio](https://multi-agent-rag-research.streamlit.app) · [Source Code](https://github.com/JeetMakadiya9/multi-agent-rag-research/tree/release/public-candidate-2)

## Overview

Multi-Agent RAG Research Studio is a research-oriented system for retrieving information, assessing supporting and contradicting evidence, and producing evidence-grounded claim-verification results.

Instead of treating retrieval-augmented generation as a single prompt to a language model, the project separates research planning, retrieval, evidence assessment, verification, and reporting into distinct components. This structure makes the research process easier to inspect, test, and reproduce.

The repository contains the research implementation, verifier variants, evaluation and experiment artifacts, automated tests, and a public Streamlit showcase.

## What the system does

- **Research planning:** Organizes research questions and identifies information needs.
- **Evidence retrieval:** Retrieves relevant material through configured search and retrieval components.
- **Evidence assessment:** Examines retrieved passages for evidence relevant to a claim, including supporting and contradicting information.
- **Claim verification:** Applies a verifier to assess claims against the available evidence.
- **Research reporting:** Presents results and associated evidence through the research workflow and interface.
- **Safeguards:** Includes quote-attestation checks and an injection guard intended to reduce the influence of malicious instructions embedded in retrieved content.
- **Reproducibility:** Maintains experiment configurations, evaluation artifacts, provenance references, and automated tests.

Exact behavior depends on the selected verifier, configuration, available evidence, and execution path.

## Architecture

The project is organized around several cooperating layers:

| Layer | Responsibility |
|---|---|
| Research orchestration | Coordinates planning, retrieval, evidence assessment, and verification |
| Retrieval and research tools | Provides access to configured information sources and retrieval components |
| Evidence assessment | Evaluates whether passages provide relevant support or contradiction |
| Verification | Applies the configured claim-verification strategy |
| Safeguards | Checks generated quotations and screens retrieved content for injection risks |
| Evaluation and reproducibility | Holds experiment runners, configurations, recorded results, provenance references, and tests |
| Research Studio interface | Exposes the workflow, recorded demonstrations, results, architecture, components, and usage guidance |

The repository's `docs/REPOSITORY_STRUCTURE.md` describes the source tree and explains why certain research scripts remain at the repository root.

## Verification in the Research Studio

The Studio currently defaults to **`arm_c_studio`**, a Studio-adapted variant of the Arm C verification approach.

The underlying approach uses evidence retrieved during research and an adjudication stage to assess claims. The Studio adaptation adds a conservative decision rule: an adjudicator-only `SUPPORTED` decision is rejected when no passage has been judged to support or contradict the claim.

The historical `arm_c` and `arm_c0` variants remain distinct from `arm_c_studio`. Earlier recorded demonstrations used the `quote_stance` verifier; their results should not be interpreted as measurements of the current Studio default.

### Safeguards and limitations

- **Injection guard:** The guard is enabled by default and can exclude a whole passage when it detects potentially malicious instructions. This can also exclude genuine evidence in the same passage.
- **Quote attestation:** Quote-generating verification paths include checks intended to ensure that quotations are grounded in their source passages.
- **Evidence dependence:** Verification quality depends on retrieval coverage, passage relevance, model judgments, and the quality of the available sources.
- **Uncertainty:** The system's output is not a guarantee of factual correctness. A verdict should be interpreted alongside its evidence and the applicable evaluation results.

## Research results and interpretation

The repository contains results from multiple historical studies and engineering investigations. These must be distinguished from the current Research Studio.

### Historical Arm C study

The recorded historical comparison reported:

| Metric | B1R | Arm C |
|---|---:|---:|
| Macro-F1 | 0.7153 | 0.7551 |
| False-support rate | 15.6% | 20.3% |

The reported Macro-F1 difference was **not statistically significant**. The result is associated with the historical evaluation data and protocol; it is not a performance estimate for `arm_c_studio`.

The repository also records an exploratory re-score and other engineering diagnostics. Those results are not interchangeable with a fresh, independently labelled evaluation.

### Phase 11B

Phase 11B contains a separate historical comparison:

| System | Macro-F1 | Evaluated examples |
|---|---:|---:|
| System B | 0.4842 | 179 |
| System C | 0.4834 | 176 |

The paired comparison did not establish a statistically significant difference. These figures describe that specific study, not the current Research Studio.

**Interpretation rule:** Do not attribute historical benchmark results to the current Studio, and do not infer current Studio accuracy from exploratory engineering diagnostics.

## Public Research Studio

The public showcase is available at:

**https://multi-agent-rag-research.streamlit.app**

The interface includes:

- Overview and project explanation
- Results and evidence
- Research workflow and recorded demonstrations
- Architecture
- Component laboratory
- Installation and usage instructions
- Use cases

The showcase is intended to explain the system and present recorded material. A displayed historical demonstration does not necessarily execute the current default verifier.

## Repository layout

The repository includes the following principal areas:

| Area | Contents |
|---|---|
| `showcase/` | Public Streamlit showcase |
| `docs/` | Architecture, repository structure, evaluation notes, and technical documentation |
| `tests/` | Automated Python and JavaScript tests |
| `webrag/` | Web research and retrieval-related components |
| `src/` | Shared source components |
| `data/` | Research data and associated artifacts |
| `evaluation/` | Evaluation-related resources |
| Root research modules and experiment scripts | Research orchestration, historical implementations, baselines, and experiment-specific runners |

Some research files remain at the repository root because recorded configurations, provenance inventories, evaluation scripts, and tests refer to their existing paths. Their location is deliberate rather than evidence that they are unused.

See [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for additional detail.

## Installation and use

The project includes multiple components with different runtime requirements. Follow the setup instructions in the repository rather than assuming that the public showcase and the complete research backend use the same environment.

1. Clone the repository and check out `release/public-candidate-2`.
2. Review the root documentation, dependency files, and relevant component-specific instructions.
3. Create and activate an environment compatible with the documented dependencies.
4. Install the dependencies required by the component you intend to run.
5. Configure the required local services and model provider, where applicable.
6. Start the relevant application using its documented entry point.

For the public showcase, begin with the instructions in [`showcase/`](showcase/). For backend and research workflows, consult the applicable component documentation and repository configuration.

Do not assume that every research workflow runs without local configuration, model availability, or external search-provider access.

## Testing

The project includes automated Python and JavaScript tests for relevant research components, application behavior, and safeguards.

The latest repository-maintenance run recorded:

- Python: **1,697 passed, 2 skipped**
- JavaScript: **54 passed**

These counts describe that reported test run, not a guarantee that every environment, external service, or research configuration behaves identically. Run the tests in the environment you use before relying on a modified checkout.

## Reproducibility and research integrity

Research claims should be interpreted using their associated evaluation protocol, configuration, data split, recorded results, and provenance information.

- Historical experiment artifacts are retained where they support reproducibility.
- Exploratory diagnostics are distinguished from formal benchmark results.
- The current Studio's accuracy must not be inferred from historical Arm C scores.
- Recorded results should not be described as independently reproduced unless that reproduction actually occurred.
- Research data and dependencies remain subject to their applicable licensing terms.

The SciFact dataset licensing terms have not been independently verified for this repository. Its inclusion should not be interpreted as a claim that every dataset artifact is cleared for redistribution.

## Limitations

- The current Studio has not been assigned an accuracy estimate from a fresh, independently labelled evaluation set.
- Retrieved evidence may be incomplete, irrelevant, contradictory, or vulnerable to errors in model judgment.
- The injection guard can remove legitimate evidence when it excludes an entire passage.
- Historical benchmark results apply only to their original evaluation settings.
- Local backend workflows may require additional configuration and running services.
- Dataset and third-party component licensing must be considered separately from the software's technical behavior.

## License and attribution

Consult the repository's license and individual dependency or dataset terms before redistributing any part of the project. Dataset licensing and redistribution permissions should not be assumed from the presence of files in the repository.

## Repository

**GitHub:** https://github.com/JeetMakadiya9/multi-agent-rag-research

**Live showcase:** https://multi-agent-rag-research.streamlit.app
