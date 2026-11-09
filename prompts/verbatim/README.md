# Verbatim prompts

These files are the exact prompt texts sent to the models. They supersede the summaries in `../prompt_templates.md`.

| File | Used by (prompt value in the metadata) |
|---|---|
| `zero_shot_prompt.txt` | `zero_shot` for all six models, `zero_shot_repeat`, and the text part of every few-shot condition except the two original models |
| `few_shot_prompt_first_models.txt` | `few_shots` for GPT-4o-mini and Gemini-2.0-Flash-Lite (the zero-shot text followed by the line `# Here are some examples:`) |
| `cot_prompt_full_wording.txt` | `cot` for GPT-4o-mini, Gemini-2.0-Flash-Lite, and Claude-Haiku-4.5 |
| `cot_prompt_condensed.txt` | `cot` for GPT-4o, Gemini-2.5-Flash, and Claude-Sonnet-4.6, and `sc5_cot` |
| `rule_free_prompt.txt` | `rule_free` |
| `zero_shot_reversed_prompt.txt` | `zero_shot_reversed` (rule blocks listed in the order 5 to 1, rule numbers and priority unchanged) |

## Few-shot exemplars

After the prompt text, each exemplar is given as `## Example i`, `Label: '<class name>'`, the image, and either `Decision: ACCEPTED` or `Rule rule<k>. Decision: REMOVE`, followed by `# Now evaluate the following image:` and the image to judge. For the image to judge, OpenAI and Anthropic requests give `The label is: '<class name>'` and `Photo: <file name>` before the image, and Gemini requests give `Label: '<class name>'`. Class names without the word leaf are given as `<class name> leaf`. The main few-shot condition uses one exemplar per category (one accepted image and one image per rule, six in total). The conditions `few_shots_k2` and `few_shots_k3` use two and three per category (12 and 18 images). Exemplar images are flagged in `manifests/split_manifest.csv` (`fewshot_exemplar_k`) and are excluded from every filtering metric.

## Sampling settings

All conditions use temperature 0 and one stateless request per image, except `sc5_cot`, which draws five responses at temperature 0.7 and keeps the majority decision, with the most frequent rule among removing responses.
