# Week 8 Qwen depth, directional siblings, and MMR

All numbers below use the frozen 80-question Legal RAG Bench test split,
Qwen3-Embedding-0.6B's saved 4,096-token-window rankings, and the same
measurement tokenizer as Weeks 6–7. The grid was specified before inspecting
test results. Its 20-question development results are in `dev_summary.csv`;
the full test grid and paired query details are in `test_summary.csv` and
`test_per_query.csv`. These are descriptive comparisons, not a newly tuned
test-set winner.

## Top-k saturation

Qwen dense children, with the existing child context assembly:

| Budget | Top-10 hits | Top-15 hits | Top-20 hits | Top-10 mean tokens | Top-20 mean tokens | Top-20 gold-text coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 53/80 | 55/80 | 55/80 | 3,170 | 3,993 | 68.8% |
| 6,000 | 53/80 | 59/80 | 60/80 | 3,203 | 5,693 | 75.0% |
| 8,000 | 53/80 | 59/80 | 60/80 | 3,203 | 6,321 | 75.0% |

At 6,000 tokens, Top-20 gives **7 additional successful questions** over
Top-10, no losses, and 8.8 percentage points more mean gold-text coverage,
for about 2,490 additional tokens per question. At 4,000 tokens the gain is
only 2 questions; at 8,000, the same 60/80 result uses another 628 tokens on
average versus 6,000. This supports a Top-10 depth limit as a major cause
of the earlier plateau. It does not establish a general optimum: the
20-question development split showed no Top-20 gain at 6,000 tokens.

## Directional sibling windows

Each policy starts from Top-10 Qwen children and adds whole siblings that
fit in the budget. `Before/after` is the maximum neighboring children on
each side of a seed. The previous `1/1` policy is the reference.

| Budget | 0/2 hits, tokens | 1/1 hits, tokens | 1/2 hits, tokens | 2/1 hits, tokens | 2/2 hits, tokens |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 51, 3,656 | 49, 3,751 | 51, 3,776 | 51, 3,791 | 52, 3,813 |
| 6,000 | 53, 4,609 | 54, 4,882 | 53, 5,074 | 55, 5,072 | 54, 5,204 |
| 8,000 | 53, 4,792 | 54, 5,205 | 54, 5,632 | 56, 5,587 | 56, 5,946 |

At 8,000 tokens, `2/1` gains two hits over `1/1` with no losses, adds about
382 tokens per question, and raises mean gold-text coverage from 67.5% to
70.0%. `2/2` gets the same 56 hits but costs about 741 more tokens than
`1/1`. At 6,000, `2/1` has two query wins and one loss, for one net hit and
about 190 more tokens. At 4,000, `2/2` adds three hits over `1/1` for about
62 more tokens, but Top-20 plain child retrieval still scores higher (55/80).
The sibling-window effects are small and can displace useful seeds under a
fixed budget; a fixed wide policy is not consistently better.

The `sibling_marginal.csv` file records per-question hit, coverage, and token
changes against child-only Top-k. For one additional second neighbor, the
paired `1/2` and `2/1` rows can be compared with `1/1`. Mean pairwise
word-set Jaccard overlap at 8,000 tokens falls from 0.192 (`1/1`) to 0.186
(`2/1`) and 0.181 (`2/2`); this is a redundancy proxy, not an evidence label.

The targeted `neighbor_steps.csv` and `neighbor_policy_per_query.csv` isolate
first and second neighbors with the **same whole-chunk packing rule** for
every step. At 8,000 tokens, adding the first preceding sibling (`0/0` to
`1/0`) gains one hit for about 1,057 tokens per question. Adding the first
following sibling (`0/0` to `0/1`) gains no hits for about 1,064 tokens.
Adding a second preceding sibling (`1/0` to `2/0`) gains two hits for about
517 tokens; a second following sibling (`0/1` to `0/2`) gains none for about
525 tokens. This directional pattern is exploratory on 80 questions and
should be checked on a second corpus before adopting an adaptive rule.

## Qwen MMR

MMR reranked each query's frozen Top-100 children using normalized Qwen
vectors and `λ × query cosine − (1−λ) × maximum selected-passage cosine`.
The candidates, context budgets, and evaluation ruler were unchanged.

| Policy at Top-10 | λ=1.0 hits | λ=0.7 hits | λ=0.5 hits | Mean overlap at λ=1.0 / 0.7 / 0.5 |
|---|---:|---:|---:|---:|
| Child only, 6,000 tokens | 53/80 | 49/80 | 31/80 | 0.227 / 0.152 / 0.136 |
| Siblings 1/1, 6,000 tokens | 54/80 | 52/80 | 41/80 | 0.191 / 0.147 / 0.135 |
| Siblings 1/1, 8,000 tokens | 54/80 | 53/80 | 44/80 | 0.192 / 0.147 / 0.136 |

At Top-10 child-only, λ=0.7 has five query wins but nine losses against
relevance-only ranking; λ=0.5 has five wins and 27 losses. Diversity lowers
surface overlap while reducing evidence inclusion. The current data do not
support using MMR at these λ values as a default.

## Interpretation and limits

The most useful next comparison is Top-20 child retrieval at 6,000 tokens
against a carefully selected expansion policy, ideally on another annotated
corpus. Legal RAG Bench has mostly one labeled gold passage per question;
these results therefore measure evidence discovery more than complete
multi-passage retrieval. FiQA provides multiple judged relevant answers but
its answers can be alternatives rather than jointly required evidence.

The child-only baseline can truncate the final chunk, while sibling policies
only add chunks that fit entirely. Comparisons between those two families
include that assembly-policy difference. Comparisons among the five sibling
policies use the same assembly rule. Pairwise overlap is lexical Jaccard over
returned chunks; it does not measure semantic duplication or factual value.
