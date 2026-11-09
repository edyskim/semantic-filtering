"""Offline regression checks for request contents, decisions, and Table 8."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from scripts import reproduce_rule_detection as rules
from scripts import run_vlm_filtering as vlm


ROOT = Path(__file__).resolve().parents[1]


class DecisionTests(unittest.TestCase):
    def test_final_decision_overrides_reasoning_and_earlier_decisions(self):
        cases = [
            ("Rule 1: Do not remove this image.\nACCEPTED", ("keep", "none")),
            ("REMOVE, Rule 1\nFinal decision: ACCEPTED", ("keep", "none")),
            ("Rule 1: Not applicable.\nRule 5: Text obscures the leaf.\nREMOVE, Rule 5", ("remove", "rule5")),
            ("ACCEPTED\nREMOVE, Rule 2", ("remove", "rule2")),
            ("Reasoning mentions remove.\n→ `ACCEPTED`\n", ("keep", "none")),
            ("**Final decision:** **REMOVE, Rule 3**", ("remove", "rule3")),
            ("```text\nREMOVE, Rule 4\n```", ("remove", "rule4")),
            ("- accepted", ("keep", "none")),
            ("Decision: REMOVE", ("remove", "")),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(vlm.parse_decision(response), expected)

    def test_unparseable_responses_are_not_silently_kept_or_removed(self):
        for response in ["", "none", "Do not remove the image.", "The image is ACCEPTED because it is clear.",
                         "REMOVE, Rule 6", "REMOVE, Rule 0", "ACCEPTED or REMOVE, Rule 1"]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                vlm.parse_decision(response)


class RequestTests(unittest.TestCase):
    def test_provider_requests_include_each_label_in_the_original_order(self):
        for provider in ["openai", "anthropic", "gemini"]:
            for class_name, expected_label in [("Tomato", "Tomato leaf"), ("Apple leaf rust", "Apple leaf rust"),
                                               ("LEAF spot", "LEAF spot")]:
                with self.subTest(provider=provider, class_name=class_name):
                    client, sdk = Mock(), Mock()
                    response = SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="ACCEPTED"))],
                        content=[SimpleNamespace(type="text", text="ACCEPTED")], text="ACCEPTED",
                    )
                    client.chat.completions.create.return_value = response
                    client.messages.create.return_value = response
                    client.models.generate_content.return_value = response
                    sdk.OpenAI.return_value = sdk.Anthropic.return_value = sdk.Client.return_value = client
                    image_part = object()
                    sdk.types.Part.from_bytes.return_value = image_part
                    modules = {"openai": sdk, "anthropic": sdk, "google": SimpleNamespace(genai=sdk),
                               "google.genai": sdk}
                    with patch.dict(sys.modules, modules), patch.object(vlm, "get_api_key", return_value="test-credential"), \
                            patch.object(vlm, "encode_image", return_value="encoded-image"), \
                            patch.object(vlm.Path, "read_bytes", return_value=b"test-image"):
                        result = getattr(vlm, f"call_{provider}")(
                            "test-model", "verbatim prompt", "data/raw/sample.jpg", class_name, 0.0,
                        )
                    self.assertEqual(result, "ACCEPTED")
                    expected = ["verbatim prompt", "\n# Now evaluate the following image:"]
                    if provider == "gemini":
                        expected.append(f"Label: '{expected_label}'")
                        request = client.models.generate_content.call_args.kwargs
                        self.assertEqual(request["contents"][:-1], expected)
                        self.assertIs(request["contents"][-1], image_part)
                    else:
                        expected += [f"The label is: '{expected_label}'", "Photo: sample.jpg"]
                        create = client.chat.completions.create if provider == "openai" else client.messages.create
                        request = create.call_args.kwargs
                        content = request["messages"][0]["content"]
                        self.assertEqual([block["text"] for block in content[:-1]], expected)
                        self.assertEqual(content[-1]["type"], "image_url" if provider == "openai" else "image")
                    self.assertEqual(request["model"], "test-model")

    def test_manifest_class_column_is_used_for_every_image(self):
        manifest = pd.DataFrame({"path": ["first.jpg", "second.jpg"], "species": ["Tomato", "Apple leaf"]})
        args = ["run_vlm_filtering.py", "--input_csv", "manifest.csv", "--output_csv", "outputs/test.csv",
                "--prompt_txt", "prompt.txt", "--provider", "openai", "--model", "test-model",
                "--image_col", "path", "--class_col", "species"]
        with patch.object(sys, "argv", args), patch.object(vlm.pd, "read_csv", return_value=manifest), \
                patch.object(vlm.Path, "read_text", return_value="verbatim prompt"), patch.object(vlm.Path, "mkdir"), \
                patch.object(vlm.pd.DataFrame, "to_csv", autospec=True) as write, \
                patch.object(vlm, "call_openai", side_effect=["Do not remove.\nACCEPTED", "REMOVE, Rule 2"]) as call:
            vlm.main()
        self.assertEqual([c.args[2:4] for c in call.call_args_list],
                         [("first.jpg", "Tomato"), ("second.jpg", "Apple leaf")])
        result = write.call_args.args[0]
        self.assertEqual(result.vlm_decision.tolist(), ["keep", "remove"])
        self.assertEqual(result.vlm_rule.tolist(), ["none", "rule2"])

    def test_missing_or_blank_labels_fail_before_any_request(self):
        args = ["run_vlm_filtering.py", "--input_csv", "manifest.csv", "--output_csv", "outputs/test.csv",
                "--prompt_txt", "prompt.txt", "--provider", "openai", "--model", "test-model"]
        for manifest in [pd.DataFrame({"img_path": ["image.jpg"]}),
                         pd.DataFrame({"img_path": ["image.jpg"], "class_name": [None]}),
                         pd.DataFrame({"img_path": ["image.jpg"], "class_name": [" "]})]:
            with self.subTest(columns=manifest.columns.tolist()), patch.object(sys, "argv", args), \
                    patch.object(vlm.pd, "read_csv", return_value=manifest), \
                    patch.object(vlm, "call_openai") as call, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    vlm.main()
                self.assertEqual(error.exception.code, 2)
                call.assert_not_called()


class RuleSelectionTests(unittest.TestCase):
    def test_selected_filters_match_released_selection_and_table_8(self):
        selected = pd.read_csv(ROOT / "summaries/kappa_selected.csv")
        detection = pd.read_csv(ROOT / "summaries/rule_detection.csv")
        columns = [f"det_rule{k}" for k in range(1, 6)] + ["valid_removed"]
        expected = {"plantdoc": [39, 87, 80, 93, 50, 9], "plantwild": [58, 93, 89, 91, 92, 16]}
        for dataset, values in expected.items():
            with self.subTest(dataset=dataset):
                selection = selected[selected.dataset == dataset][["model", "prompt"]].drop_duplicates()
                self.assertEqual(list(selection.itertuples(index=False, name=None)), [rules.SELECTED[dataset]])
                model, prompt = rules.SELECTED[dataset]
                row = detection[(detection.dataset == dataset) & (detection.model == model)
                                & (detection.prompt == prompt)]
                self.assertEqual(len(row), 1)
                self.assertEqual([int(f"{100 * x:.0f}") for x in row[columns].iloc[0]], values)


if __name__ == "__main__":
    unittest.main()
