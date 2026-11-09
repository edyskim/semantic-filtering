# Models, identifiers, and access dates

| Model | API identifier | Access dates | Conditions |
|---|---|---|---|
| GPT-4o-mini | `gpt-4o-mini` | 6 to 8 Aug 2025, 3 Nov 2026 | zero-shot, few-shot, CoT (2025); rule-free, repeated zero-shot (2026) |
| Gemini-2.0-Flash-Lite | `gemini-2.0-flash-lite` | 6 to 8 Aug 2025 | zero-shot, few-shot, CoT |
| GPT-4o | `gpt-4o` | 30 May 2026, 3 Nov 2026 | zero-shot, few-shot, CoT (May); rule-free, repeated zero-shot (Nov) |
| Gemini-2.5-Flash | `gemini-2.5-flash` | 30 to 31 May 2026, 3 Nov 2026 | zero-shot, few-shot, CoT (May); few-shot with correctly labeled exemplars (Nov) |
| Claude-Sonnet-4.6 | `claude-sonnet-4-6` | 30 to 31 May 2026, 3 Nov 2026 | zero-shot, few-shot, CoT (May); rule-free, self-consistency (Nov) |
| Claude-Haiku-4.5 | `claude-haiku-4-5-20251001` | 30 May 2026, 3 Nov 2026 | zero-shot, few-shot (May); CoT, rule-free, self-consistency, ablations, repeated zero-shot (Nov) |

Apart from Claude-Haiku-4.5, the identifiers are provider aliases. In November 2026 the OpenAI aliases resolved to `gpt-4o-mini-2024-07-18` and `gpt-4o-2024-08-06`. Gemini-2.0-Flash-Lite has since been withdrawn by its provider. Because the models can change or disappear, every decision is released in `metadata/vlm_filtering_metadata.csv`, and all filter-level analyses can be recomputed from it without querying any model.

The original few-shot runs of the two Gemini models displayed the accepted exemplar with the decision REMOVE. These rows are kept in the metadata with `in_summaries = False`. Gemini-2.5-Flash was re-run with correct exemplars (`few_shots_fixed`), which replaces the original run in all summaries. Gemini-2.0-Flash-Lite could not be re-run.
