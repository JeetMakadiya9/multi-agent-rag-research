# webrag — internet research and fact verification

## Launch

```
.venv\Scripts\python.exe -m webrag.server                 # web interface at http://127.0.0.1:8765
.venv\Scripts\python.exe -m webrag "research question"     # command line, research mode
.venv\Scripts\python.exe -m webrag --verify "a claim"      # command line, verification mode
```

Requirements: a running local Ollama with `qwen3:4b` (no other model is substituted), the cached
`all-MiniLM-L6-v2` embedding model and MS-MARCO cross-encoder used by `rag.py`, and internet access. No local
documents are needed. Every run is saved to `outputs/webrag/runs/<time>-<slug>/run.json` (sources, extracted text,
passages, stances, stages).

## Research Studio (web interface)

`python -m webrag.server` serves the studio at http://127.0.0.1:8765. Options: `--port`, `--model` (default
`qwen3:4b`), `--verifier` (Studio default `arm_c_studio`: Arm C agentic verification, Studio-adapted variant, see
below; `arm_c` is the historical Arm C, `arm_c0` the historical arm C0, `quote_stance` the previous default and legacy
fallback; `quote_stance_ranked` and `quote_stance_adjudicated`, the last not recommended, are alternatives that are not
Arm C), `--injection-guard` (Studio default `exclude`: retrieved passages with instruction-like text are withheld from
every model call; `flag` records them only; `off`; the screen's evaluation found no benign flags on two held-out
corpora but misses paraphrased attacks, and its pre-registered gate had recommended keeping it opt-in, see
`docs/reverse_engineering/EVALUATION_RESULTS.md`), `--quote-attestation` / `--no-quote-attestation` (Studio default:
on whenever the verifier quotes; records where each verifier quote occurs in its passage, never changes a verdict; see
"Correctness checks" below), `--docs` (document folder, default `data/documents`) and
`--obsidian-vault PATH` (or `WEBRAG_OBSIDIAN_VAULT`) to allow Markdown export into an Obsidian vault. Without a flag,
`--verifier`, `--injection-guard` and `--quote-attestation` take `WEBRAG_VERIFIER`, `WEBRAG_INJECTION_GUARD` and
`WEBRAG_QUOTE_ATTESTATION` from the environment or `.env` (`webrag.model_service.studio_options`); an invalid value
stops the start, and `/api/status` reports each option with where it came from (`option_sources`), plus
`verifier_description` and `verifier_kind`. A run request that names `verifier`, `injection_guard`,
`quote_attestation`, `followup_rounds` or `planner` is refused (400): these are fixed when the Studio starts. The
command-line tool (`python -m webrag`) and the evaluation harness keep `Settings`' own defaults (`quote_stance`, guard
off, attestation off), so their earlier results stay reproducible.

* **Where things live.** Saved runs: `outputs/webrag/runs`. Documents: the `--docs` folder. Runs in progress,
  document answers and the document index: server memory only. Preferences (theme, text size, motion, research
  defaults, technical details) and the local profile (display name, initials): this browser's `localStorage`, keys
  `studio.preferences` and `studio.profile`. The profile is not an account; nothing is sent anywhere.
* **Integrations** are reported from real configuration only: Ollama (status, installed models and a connection test,
  all local GET requests), the research sources (switches, SearXNG address, feeds and a test search; see Deep
  research below), the document folder, the Obsidian
  vault (validated: it must contain `.obsidian`; the studio writes new notes into `<vault>/Research Studio` with
  exclusive creation, never reads or overwrites notes) and Markdown/JSON export of saved runs. Cloud model APIs are
  not available: `docqa.llm.OllamaChat` accepts local hosts only and refuses `:cloud` models.
* **What leaves the computer.** Research and verification send search queries to the configured providers and fetch
  the result pages. The model, the document assistant and saved reports stay local. Keys are read from the
  environment; the interface shows only whether each is set.
* **Cancelling.** A queued run can be withdrawn and never starts; a started run cannot be interrupted safely, so the
  page stops following it and it still finishes and is saved.

**Local use only.** The server binds to 127.0.0.1, answers only requests whose `Host` is `127.0.0.1` or `localhost`
(against DNS rebinding), refuses cross-origin writes and sends a strict Content-Security-Policy, but it has no
authentication. Before any network or online deployment it needs: user accounts and server-side authorization (a
browser profile is not authentication), TLS, CSRF tokens instead of the Origin check alone, per-user storage of runs,
documents and preferences, isolation and limits for the job queue, upload scanning and quotas, a secrets store
instead of environment variables, the Obsidian export disabled or made per-user, rate limiting, and audit logging.

## Network safety

Every research request goes through `webrag.net.PoliteSession`. Each URL and every redirect hop (and each robots.txt
fetch) must be http(s), must not name a local host or a numeric address in a non-standard form (`2130706433`,
`0x7f.1`, `127.1`), and every address its name resolves to must be globally routable: loopback, private, link-local,
CGNAT (100.64.0.0/10), reserved, multicast and unspecified addresses are refused, and one refused address refuses the
host. On networks with DNS64 (common on mobile and IPv6-only networks), IPv4-only hosts also resolve to NAT64
addresses in the RFC 6052 well-known prefix `64:ff9b::/96`; such an address is judged by the IPv4 address it embeds
(public: allowed; loopback, private, metadata and the like: refused). The local-use prefix `64:ff9b:1::/48` and
operator-specific prefixes are not decoded and stay refused. TLS certificates are always verified. A refusal is
reported as `blocked_host` or `proxy_policy`, and a provider refused this way is shown as "blocked by network policy",
not as failed; unreachable hosts are reported as `dns`, `tls`, `proxy`, `timeout` or `network`.
Connections are pinned: the request connects to an address that passed the check, never to a fresh DNS answer
(DNS rebinding), while TLS still verifies the certificate chain and the original host name (sent as SNI) and the
Host header carries the original host. A failed connection moves to the next checked address (at most 4); a TLS
failure is never retried; an answer is re-checked after 5 minutes. Requests through an accepted proxy
(`WEBRAG_ALLOW_PROXY=1`) are not pinned, because the proxy resolves destinations itself.

## Configuration (optional)

| Variable | Effect |
|---|---|
| `SEARXNG_URL` | adds a self-hosted SearXNG (general web, government and institutional pages); e.g. `http://127.0.0.1:8888` |
| `SEARXNG_ENGINES`, `SEARXNG_CATEGORIES` | optional SearXNG `engines` / `categories` parameters (comma-separated) |
| `BRAVE_API_KEY` | adds the Brave Search API (paid tiers; optional) |
| `OPENALEX_API_KEY` | free OpenAlex key, sent as an `Authorization: Bearer` header (never in URLs or run records) |
| `WEBRAG_PROVIDERS` | replaces the default provider set, e.g. `wikipedia,openalex,arxiv,europepmc,searxng` |
| `WEBRAG_CONTACT` | contact string added to the User-Agent sent to APIs and sites |
| `WEBRAG_ALLOW_PROXY` | `1` lets research requests go through a proxy from the environment or system settings; without it they are refused (`proxy_policy`), because the proxy resolves destinations itself, outside the address check |

Providers: `wikipedia` (default for the command-line pipeline only: the studio never uses it, see Source policy below),
`openalex`, `arxiv` (default, keyless), `searxng` and `brave` (default when configured),
`europepmc` (opt-in, keyless: biomedical and life-science articles with abstracts and open-access full text through
its full-text API), `crossref` (opt-in, keyless: DOI metadata, abstracts when deposited, a PDF link only under a
Creative Commons licence) and `feeds` (RSS/Atom feeds you list in the studio). Naming an unknown or unconfigured
provider is an error. Every run records `provider_status`
(queries, failures by code, hits) and `source_diversity` (domains, publishers, independent origins), and the report
lists each failing provider once.

**OpenAlex.** Without a key, requests share a daily budget per network address; when it is used up OpenAlex answers
HTTP 429 until midnight UTC, the client stops calling it for the rest of the run, and the report says so. A free key
has its own budget: create one at https://help.openalex.org/api/authentication/ and set `OPENALEX_API_KEY`.

**SearXNG (recommended for general web coverage; free, self-hosted).** Not tested on this machine (no Docker). With
Docker, following https://docs.searxng.org/admin/installation-docker.html:
`docker run -d -p 8888:8080 -v "%CD%\searxng:/etc/searxng" -e "BASE_URL=http://127.0.0.1:8888/" searxng/searxng`,
then in `searxng\settings.yml` add `json` to `search.formats` (the client reports HTTP 403 otherwise) and restart.
Set `SEARXNG_URL=http://127.0.0.1:8888`. Only that configured origin may be a local address; result pages are still
fetched under the normal rules (robots.txt, no private addresses, every redirect hop re-checked).

Without SearXNG or Brave, current news and general web pages are out of reach (except through feeds you list), and
every report says so.

## Deep research (studio runs)

Runs started from the studio use the bounded multi-round engine in `webrag/deep.py`; the command line, the evaluation
and `Orchestrator.run(text)` without options keep the single-pass pipeline below unchanged (`Settings.depth=None`).
A run is `Orchestrator.run(text, mode, options={"depth", "providers", "categories", "recency"})`; the server checks the
options (`POST /api/run`) and accepts only providers that are active.

**Search configuration.** Switches, the SearXNG address/engines/categories, the feed list and an optional Crossref
contact email are stored in `outputs/webrag/studio-config.json` (not versioned) through `GET /api/providers` and
`POST /api/providers/config`. Keys are never stored there: unknown fields (an API key, for example) are refused, and
`BRAVE_API_KEY` / `OPENALEX_API_KEY` stay in the environment, reported only as set or not set. `WEBRAG_PROVIDERS` and
`SEARXNG_URL` take precedence over the file. `POST /api/providers/<id>/test` sends one neutral query ("renewable
energy") to one provider, only when the user presses Test search. The catalog distinguishes implemented, configured,
switched on and active; each run records which providers it actually used, per round.

**Source policy: Wikipedia is not an evidence source.** In the studio the Wikipedia provider cannot be switched on
(`config.validate`), is never built from the studio configuration even if `WEBRAG_PROVIDERS` names it
(`search.default_providers` with a configuration), is removed from the studio's single-pass fallback
(`server.studio_orchestrator`) and from any deep run it is passed to (`deep.DeepResearch`), and cannot be tested.
Pages on `wikipedia.org` (any subdomain) that another provider returns are dropped before they are fetched, and a page
that redirects there is dropped after it (`WebResearcher.gather` in deep mode), so they never become sources: they
cannot be indexed (`deep.is_evidence`), cited (`deep.check_citations` flags such a passage), count towards coverage or
corroboration, or appear in the bibliography. Each run reports how many such results were left out. Wikipedia is not
used for discovery either. The command-line pipeline and the evaluation keep their original provider defaults.
Mirrors of Wikipedia on other domains are not recognised.

**Depths** (upper limits; "searches" are query × provider calls; time is when searching stops, drafting and
verification still finish):

| | rounds | queries/round | searches | sources | full-text attempts | sub-questions | findings | counter-evidence checks | minutes |
|---|---|---|---|---|---|---|---|---|---|
| quick | 1 | 4 | 10 | 10 | 4 | 3 | 6 | 0 | 8 |
| standard | 2 | 6 | 24 | 16 | 8 | 4 | 8 | 2 | 15 |
| deep | 3 | 8 | 44 | 24 | 12 | 5 | 8 | 3 | 30 |

**Loop.** `agents.SourceAwarePlanner` writes sub-questions and queries tagged with a source type (encyclopedia,
scholarly, web, news, official, podcast); each query goes to the active providers of that type (falling back to the
input's keywords when the model's plan is unusable). After each round, a coverage check ranks passages for every
question: *strong* means passages from two or more publishers established as independent (see Independence below),
including full text. Searching stops when every
question is strong (`retrieval_coverage_met`; called `coverage_sufficient` in deep-research-v1 records), or on
`time_budget`, `round_limit`, `search_budget`, `source_budget`, `providers_failing`, `little_new_evidence` (a follow-up
round found no new relevant passage) or `no_new_queries`; otherwise the model writes follow-up queries for the open
questions, preferring source types not yet used. The coverage ledger in the record lists, per question, the queries,
sources found and read, supporting and contested statements, a status (answered, answered with disagreement, partially
answered, contested, unanswered, not reached), the reason and a next step. The stopping record keeps apart: `reason`
(the condition that stopped searching), `also_reached` (other conditions that held at the same time, such as
`round_limit` at Quick depth), `retrieval_coverage` (whether the retrieved passages met the retrieval check, which is
not an answer), `answer_coverage` (questions answered by verified statements, and the open ones) and `completeness`
(complete, partial, budget-limited, evidence-limited, access-limited, provider-limited). A retrieval stop is complete
only when every question is answered; otherwise it is partial and its detail names the open questions. The run status
(below) is separate.

**Query allocation and the query ledger** (deep-research-v3). When a round has more planned queries than its
`queries_per_round`, `deep.allocate_queries` keeps the user's wording, reserves a quarter of the places (at least one,
for rounds of three or more) for counter-evidence, and takes each subquestion in turn before any is searched twice;
earlier versions kept the first N, which dropped the planner's counter-evidence queries at every depth. Searches are
routed round-robin (every query gets its first provider before any gets a second). `data["queries"]` lists every
planned query with `executed` (zero results still counts), `failed`, `skipped` (with the budget that caused it) or
`no_provider`; a counter-evidence query that did not run is never reported as sought. `data["provider_coverage"]`
gives each provider's state in the run: searched, failed, not used, not selected, not configured, off or excluded.

**Candidates, selection and full text** (deep-research-v3). Deep research asks each provider for
`ProviderInfo.candidates` results per search (OpenAlex, Crossref, SearXNG, Brave 10; arXiv, Europe PMC, feeds 8) in a
single request, without paging; the source budget still caps what is read, and the command line and evaluation keep
`Settings.per_query = 3`. `WebResearcher` (deep mode only) then reads the best-scoring result of each query first, and
the rest by `researcher.candidate_score` (keyword relevance to the queries, readable content, provider rank) with a
small penalty for a venue or site already chosen. Full-text attempts are granted after every selected source is known,
to the highest-ranked ones with an open-access copy, instead of to whichever download thread asked first; the others
record why they stayed abstracts. The ranking is a reading order, never evidence. Follow-up queries are written with
the gap that keeps each question open (no relevant passage, abstracts only, one publisher, independence unknown).

**Claims mode** (deep-research-v3). A verification request may hold up to 8 claims, one per line; each is judged on
its own by the configured verifier against the passages retrieved for it, and the run's outcome is `CLAIMS_ASSESSED`
(there is no overall verdict). The request counts as answered only when every claim has a decisive verdict. One line
is one claim, exactly as before. Claims are not split into subclaims automatically, and a quotation or URL given as
context is not fetched: write each checkable proposition as its own line.

**Gate 1 repairs** (deep-research-v4, after the first live v3 run). Corroboration counts a supporting source only when
one sentence of its passage contains at least 60% of the finding's content words (`deep.support_check`); a passage
that only defines a concept or touches the topic does not corroborate, and is listed in the review as not counted.
This lexical screen can only withhold corroboration; it is not proof of entailment, and the verifier's verdicts are
unchanged. New cautions `qualifier_dropped` (a quantifier such as "many", the modal "can", or a scope such as "on two
datasets" in every best-matching supporting sentence but not in the finding) and `attribution_dropped` (the source
reports a named work's result, the finding does not say so); a statement resting on another work's result reported by
the source is listed as a secondary report. Summary sentences must keep each cited finding's qualifiers, scope and
attribution or are dropped. When the planner tags every query with the same subquestion, `deep.infer_subquestions`
links queries to subquestions by the words only one subquestion uses, and allocation gives each subquestion one query,
the least covered first; a subquestion left without a search of its own is named in the review. Results with an
abstract below `researcher.RELEVANCE_FLOOR` (0.7 of a query's keywords; calibrated on one run) are not read.
Full-text attempts are reported as attempts, successes and failures. Passages that begin with the previous sentence's
full stop are classified from after it. All of these are lexical heuristics with known misses: paraphrased qualifiers
are not recognised, and a faithful paraphrase can be flagged.

**Answer eligibility** (deep-research-v5, after the v4 live smoke test). A statement the verifier judged SUPPORTED but
whose supporting passages all fall below the 60% coverage of the support check gets the caution `support_unmatched`:
it stays listed with its evidence and the reason, and like every cautioned statement it is not offered to the direct
answer or summary. It is not marked false. Scope phrases may contain punctuation ("on a large-scale, class-imbalanced
transaction dataset"). Direct-answer and summary text longer than 600 characters is shortened between sentences and
ends with " […]" (or, for one over-long sentence, at a word with "…"); the qualifier checks run on the shortened text.
Word matching cannot see a changed comparison (one graph model against other graph models written as graph models
against conventional methods) or a result from one model and dataset generalised to a whole class of methods.

**Evidence review** (`deep.review`, deep-research-v3). A deterministic list, per statement (`C1`, `C2` ... in the record
and the report), of what limits its support (cautions, abstract-only support, corroboration not established,
contested, failed checks) and of planned work not done (queries not run, counter-evidence not sought, open questions),
each with the passages or queries it refers to and what was done (left out of the answer, needs evidence, unresolved).
It restates the recorded checks; no model is asked and it is not an independent verification.

**Evidence policy** (`deep.EVIDENCE_POLICY`, recorded in every deep run). Only FULL_TEXT and ABSTRACT_ONLY sources are
indexed as evidence; SNIPPET_ONLY results and METADATA_ONLY records (for example a DOI record with no abstract and no
open copy) are used to discover sources only. Findings are drafted per question (`Synthesizer` prompt v3) and judged by the
same verifier as the single-pass pipeline; its verdicts are never changed. A descriptive label is added to each:
corroborated (two or more publishers established as independent, one in full text, nothing against), independence
unknown (several sources whose independence the metadata cannot establish), single source (one publisher, work or
text), abstract only, contested, contradicted, not established, unverified. No numerical confidence score is computed.
Every passage behind a verdict is checked to come from a recorded, readable source that the source policy accepts and
an existing page or time window. At Standard and Deep depth, supported findings whose publishers are not established
as independent are searched for counter-evidence and verified again; both verdicts are kept. The direct answer and
summary (`agents.summarize`) restate verified findings only: sentences that cite none, or add a number not in the
findings they cite, are dropped and counted.

**Cautions** (`deep.cautions`, deep-research-v2). Fixed text checks add warnings to verified findings; they are not
verdicts and not proof of entailment, and the verdict is never changed. A cautioned finding stays listed with its
warning and evidence but is left out of the direct answer and summary. `fragment_subject`: the finding's best-matching
supporting sentence is the cut-off start of a passage that begins mid-word or mid-sentence. Passage starts are
classified from the source's own page text (`deep.passage_boundary`): mid-word or mid-sentence only when the text
before the passage confirms it (at a page break: a line ending in a hyphenated word part whose joined form occurs in
the source), complete when a sentence starts there, and uncertain otherwise (no page text, the passage not found, a
possible heading, an unclear page end). Running headers and footers repeated across pages are skipped. Chunk
boundaries (from `rag.py`) are not changed, and the verifier still judges the passage as retrieved, so a wrong subject
can still be verified; the caution only keeps it out of the answer. `hedge_dropped`: every best-matching supporting
sentence is hedged (may, might, could, suggests, appears, is associated with ...) but the finding is not; supporting
sentences of mixed strength give no caution. `causal_added`: the finding states cause and effect (causes, leads to,
because, due to ...) that no best-matching supporting sentence states, as when a source reports an association.
Negation in the finding or its supporting sentence is not judged (`fidelity_check` records "not assessed"). These are
lexical cues: they miss paraphrased hedges and causes and can flag a causal word used in another sense. The synthesis
(v3) and summary prompts also ask the model to keep hedges and not add causes; prompts cannot guarantee it. Summary
sentences that add a causal word none of their cited findings uses, or drop a hedge all of them carry, are dropped and
counted like those with an unsupported number. `number_unsupported` (deep-research-v3): the finding states a number that
appears in none of its supporting passages (thousands separators are ignored; units, rounding and recomputed values
are not judged).

**Independence** (`deep.independence`) is judged from recorded metadata and the sources' own retrieved text, not from
how different their wording is. Near-copies (`Source.origin`, 60% 8-word shingle containment) still count once and
still limit redundant passages, but distinct texts are not independent by that alone. Two sources are the same
publisher when they are copies, the same work (DOI, arXiv, PMID, PMCID), two versions of one work (scholarly, same full
title), on the same site (last two host labels, three under `co.uk`-style country domains) or carry the same recorded
publisher name; scholarly works are the same team when they share an author (surname and initial). A source whose
retrieved text contains another's DOI, arXiv ID, PMID, PMCID, URL or full title (five words or more) is *dependent* on
it (deep-research-v2). Pages are independent only with different sites *and* different recorded publisher names and
no such citation. Scholarly works are independent only when their author lists share no author, both were read in full
text with a detectable reference-list heading, and neither cites the other: different authors alone are not enough.
Anything else (a dependent pair, a page without a recorded publisher name, a work without authors or without a
checkable reference list, a news page about a study) is "independence unknown", which never counts as corroboration.
Each assessment records the relation and reason for every pair. This is a heuristic over the retrieved text, not a
citation-graph search: it misses citations in text that was not retrieved (abstracts have no reference list),
indirect chains and references it cannot match. The domain alone is never taken as a publisher name, and different
domains are never assumed to be independent. Records keep the original `independent_support` count (distinct texts)
for compatibility.

**Run status.** FAILED means an execution failure: no accepted provider, every search failed, the model failed or its
synthesis was unusable, or an unexpected error. A run whose searches ran but found nothing readable is COMPLETE (PARTIAL
when some searches failed) with the outcome Insufficient evidence and the reason in `error`, as in the single-pass
pipeline; nothing is presented as an answer.

**Content.** Pages: main-content extraction (`<main>`/`<article>` when substantial), charset from the header, BOM or
`<meta>`, then detection; short login, paywall, cookie-wall and bot-check pages are not read: the source is recorded as
not retrieved with that reason (a search snippet, if any, is kept for discovery only), and nothing is bypassed. PDFs: page count, pages without text, scanned (image-only) detection. Scholarly works: abstract,
then Europe PMC full text, then the best open-access location from OpenAlex (by DOI), within the full-text budget.
Feeds: RSS/Atom matched by keyword; podcast episodes with a published transcript (Podcasting 2.0: WebVTT, SRT or JSON)
are read from it and quoted by time window (`0:01–1:00`). Audio and video are never downloaded or transcribed, and the
report never says media were watched or listened to. A recency choice (past year, past 5 years) uses the provider's
own date filter where it has one and drops dated results outside the range; undated results are kept and the report
says so.

**Report.** Direct answer and summary with finding references, findings by question with verdict and evidence label,
counter-evidence changes, disagreements, evidence quality and source diversity (counts by access level and source
type, distinct texts, publisher groups, domains), unresolved questions, limitations, method and stopping reason, and a
bibliography with authors, identifiers (DOI, arXiv, PMID, PMCID), page or timestamp. The studio shows the same, plus a
Coverage tab, per-round queries and providers, budget used versus limits, and (with technical details on) per-round
provider status and citation checks.

## Pipeline

| Agent | Module | Role |
|---|---|---|
| A Planner | `agents.Planner` | subquestions and ≤4 search queries (one for contrary evidence); never states facts. Opt-in `Settings(planner="perspective")`: the same limit, but each query is written for a research direction (supporting, challenging, context: methodology, limitations, alternatives) and only relevant directions are used |
| B Web researcher | `researcher.WebResearcher`, `search`, `net`, `extract`, `dedupe` | real searches, polite fetching (timeouts, size caps, per-host spacing, robots.txt for pages), HTML/PDF extraction, URL/DOI/arXiv de-duplication, near-copy detection; records FULL_TEXT / ABSTRACT_ONLY / SNIPPET_ONLY / FAILED per source |
| C Evidence | `evidence.EvidenceStore` | per-source hybrid retrieval through the unchanged `rag.py`, global ordering by cross-encoder score, at most 2 passages per origin |
| D Verifier (Studio default) | `docqa.agentic.StudioArmCVerifier` (`arm_c_studio`) running the agentic verification study's frozen arm-C code (`evaluation/experiments/agentic_verify/arms.py`, `prompts.py`, `llm.py`, byte-identical, settings from its `FROZEN_CONFIG.json`) through the Studio's model client | one verifier call (prompt v1) judges up to 8 passages, each SUPPORTS / CONTRADICTS stance needing an exact quote; unanimous evidence decides; conflicting evidence goes to the adjudicator (prompt v2, at most 6 passages shown); when no passage took a side, the planner (v0) may run at most 2 searches of this run's evidence store (`EvidenceStore.rank`, 4 new passages each, judged by the same verifier) in 3 turns; an unusable adjudication falls back to the rank rule (RANK_TIEBREAK). An adjudicated conflict is sub-state ADJUDICATED (cautioned, kept out of the answer, both sides listed). A decision without any passage for or against is declined (`arm_c_studio`) or accepted as ADJUDICATED_UNSIDED (`arm_c`). Passages found by search are recorded with `found_by_search`, and every claim records `verifier` and `verifier_trace` (steps, adjudication, usage). Model-service failures give SERVICE_ERROR, unusable verifier output ASSESSMENT_FAILED. `arm_c0` is arm C0 (no planner or search). With quote attestation, the verifier's quotes are captured from its replies and attested (record only). A planner search that fails, or has no evidence store, fails closed (ASSESSMENT_FAILED). **Not evaluated on the Studio's web evidence**: Arm C's 0.7551 macro-F1 is a historical SciFact result on previously used claims (not significantly above the rank baseline, p = 0.081), and its adjudicator raised false support there (13/64 against 10/64) |
| D Verifier (previous default, CLI and evaluation harness default) | `docqa.verify.QuoteStanceVerifier`, `CheckedStanceVerifier` (`--verifier checked_stance`, v4: the v2 judgement plus a focused yes/no check of every SUPPORTS and CONTRADICTS stance on the real passage sentence, compound claims split into faithful facts), `PropertyStanceVerifier` (`--verifier property_stance`, high precision) or the frozen Experiment B verifier (`--verifier frozen_b`) | per passage: an exact quote, whether it concerns the same subject and property, and SUPPORTS / CONTRADICTS / NEUTRAL; code rejects quotes not in the passage and support that lacks the claim's numbers; SUPPORTED, CONTRADICTED or INSUFFICIENT (conflicts and failed checks are INSUFFICIENT) |
| E Synthesis | `agents.Synthesizer` (prompt v2: one fact per finding, details only from passage text, no dates in passage labels), `report` | cited findings from passages; the report lists as findings only claims the verifier marked SUPPORTED, with the sources whose passages supported them |
| Follow-up (opt-in) | `followup.run_followup`, `Settings(followup_rounds=1)` or `python -m webrag.followup --run FOLDER` | off by default; at most one round of at most 2 queries, only for findings judged INSUFFICIENT for lack of decisive evidence (never for conflicts or failed checks); original verdicts and sources are kept in the record, new sources are marked `followup_round: 1` |
| Rank tie-break (experimental opt-in) | `docqa.ranked`, `--verifier quote_stance_ranked` | the `quote_stance` judgement; only when grounded stances conflict, the first passage in evidence order with a SUPPORTS or CONTRADICTS stance decides (claim check: retrieval rank; research finding: its cited passages first). No extra model call. Recorded as sub-state RANK_TIEBREAK with the disagreement kept; a supported statement gets the `rank_decided_conflict` caution and stays out of the answer and summary. Origin: the SciFact agentic verification study, where the rule raised macro-F1 over abstaining on conflicts without raising false support; **not evaluated on the Studio's web evidence** |
| Conflict adjudication (experimental opt-in) | `docqa.adjudicate`, `--verifier quote_stance_adjudicated` | the `quote_stance` judgement; only when grounded stances conflict, an adjudicator call (asked in both passage orders) may decide one side, citing a validated passage and quote; otherwise INSUFFICIENT / CONFLICTING stays. Recorded as sub-state ADJUDICATED with the disagreement kept. **Not recommended**: in exploratory runs it raised false support from 0.133 to 0.211 — see `evaluation/webrag/ADJUDICATED_VERIFIER.md` |

## Correctness checks (2026-10-23)

These change no verdict rule (`docqa/verify.py` is frozen and byte-identical). Tests: `tests/test_numeric_tokens.py`,
`tests/test_quote_attestation.py`, `tests/test_source_status.py` (modules alone) and `tests/test_numeric_grounding.py`,
`tests/test_quote_attestation_integration.py`, `tests/test_injection_guard.py` (through the application).

* **Numbers in the direct answer and summary** (deep research; on by default, a bug fix). A summary sentence is kept
  only if every number it states appears as the same whole token in the findings it cites (`webrag/numeric.py`).
  Before, digits inside another number counted ("5%" passed against "25%", "2" against "2024"). The selected contract
  compares tokens as written — a conservative choice, not a statement that the values differ: "1,500" ≠ "1500",
  "1.50" ≠ "1.5", "1,5" ≠ "1.5", "-5" ≠ "5" (the Unicode minus equals "-"); units and
  percent signs are not compared; ranges and dates are their parts; scientific notation is not interpreted. A
  reformatted number therefore drops the sentence (counted in `summary.dropped`) rather than letting a figure through.
  The verifier's own claim/passage number check and the `number_unsupported` caution keep their existing rules.
* **Quote attestation** (record only; on by default in the Studio for quoting verifiers, opt-in on `python -m webrag`:
  `--quote-attestation`,
  `Settings(quote_attestation=True)`). The verifier's quote check tolerates reordered words, a dropped "not" and
  matches inside longer words. With this option every judged passage also records where the verifier's final quote
  occurs in it (`evidence_core/span_attestation.py`: LOCATED / AMBIGUOUS / UNATTESTED / EMPTY, EXACT or CANONICAL
  tier, offsets into the passage, a negation cue), next to the verifier's own check, under
  `claims[].passages[].quote_attestation` in `run.json` and in a "Quote attestation" table in the Markdown report.
  It describes text position only: it rejects nothing, changes no stance or verdict, and a located quote can still
  fail to support a claim. Refused with an error for verifiers that do not quote (`frozen_b`, `property_stance*`).
  Not recorded: the adjudicator's own deciding quote in `quote_stance_adjudicated`. Runs without the field load and
  render as before.
* **Retraction status** (offline contract only; NOT AVAILABLE in research runs). `webrag/source_status.py` defines the
  states RETRACTED, NOT_RETRACTED, CONFLICTING, LOOKUP_FAILED and UNKNOWN and how provider assertions combine:
  NOT_RETRACTED needs an explicit assertion from a provider that covers retractions; missing metadata, no DOI or no
  lookup is UNKNOWN; errors and malformed records are LOOKUP_FAILED; an assertion counts only for the DOI it names (a
  notice, preprint or other version does not transfer); corrections are notes. No provider is queried, no source
  record carries a status, and it does not affect any verdict, label or corroboration rule. Before live use, an
  adapter must be validated against real responses for: which provider field states a retraction and whether its
  absence means "not retracted"; DOI coverage per source type (many web and preprint sources have none); how a
  retraction notice names the retracted work; preprint/journal version links; data freshness; and timeouts,
  rate limits and malformed responses. Duplicate search hits are still merged first-wins (`researcher.py`), so
  conflicting metadata from different providers is not recorded.

## Evaluation

```
python -m webrag.evaluate collect --dataset evaluation/webrag/claims_dev.jsonl --snapshot dev-20261018
python -m webrag.evaluate score --dataset evaluation/webrag/claims_dev.jsonl --snapshot dev-20261018 \
    --verifier frozen_b --run-name dev-baseline
```

`collect` freezes the searched and extracted sources per claim; `score` runs offline on that snapshot, so verifiers
are compared on identical evidence. Datasets are hash-frozen in `evaluation/webrag/FROZEN.json`. Method, label
sources and metric definitions: `evaluation/webrag/README.md`.

Measured on the held-out claim set (30 claims, 10 per label, one snapshot, identical passages for both verifiers):

| | frozen Experiment B verifier | `quote_stance` v2 (CLI default; the Studio's previous default) |
|---|---|---|
| accuracy | 20/30 | 26/30 |
| macro-F1 | 0.581 | 0.864 |
| contradiction recall | 1/10 | 7/10 |
| false SUPPORTED | 3 | 1 |
| mean seconds per claim | 69 | 26 |

These are small-sample results on short encyclopedic claims, not a general accuracy figure. The adoption rule was
written before scoring (`evaluation/webrag/DECISION_RULE.md`).

Follow-up on property confusions (`evaluation/webrag/DECISION_RULE_V3.md`): a new held-out set of 36 claims with
"trap" values that belong to a different property of the same subject (e.g. a launch year offered as the sinking
year) and compound claims. On it v2 scored 25/36 (macro-F1 0.69) with 3 false SUPPORTED verdicts. The candidate
`property_stance` (v3c: per-fact checks with a yes/no confirmation on the real passage sentence) made no false
SUPPORTED verdicts but recognised only 5 of 12 true claims (24/36, macro-F1 0.66) and was not adopted. Use
`--verifier property_stance` when wrongly confirming a claim is costlier than failing to confirm one.
