# Prompt templates

## Zero-shot template

```text
Task: Image data cleaning for plant disease classification.

You are given an image and its class label.

Apply the expert-defined semantic filtering rules in priority order:
1. Plant-class mismatch.
2. Main subject is not a leaf.
3. Illustration or artificial graphic.
4. Document or presentation capture.
5. Obstructive or excessive text.

If any rule is clearly satisfied, output:
REMOVE, Rule [number]

If no rule is clearly satisfied, or if the case is ambiguous, output:
ACCEPTED
```

## Few-shot template

```text
Task: Image data cleaning for plant disease classification.

Use the expert-defined semantic filtering rules and the provided examples to decide whether the target image should be accepted or removed.

For each example, consider the label, image content, expert decision, and rule attribution.

For the target image, output exactly one of:
REMOVE, Rule [number]
ACCEPTED
```

## Chain-of-thought template

```text
Task: Image data cleaning for plant disease classification.

Evaluate the image using the expert-defined semantic filtering rules in priority order.

For each rule, briefly state whether it is applicable. If a removal rule is clearly applicable, stop at the first applicable highest-priority rule.

End with exactly one final decision:
REMOVE, Rule [number]
ACCEPTED
```
