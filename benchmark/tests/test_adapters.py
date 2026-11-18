import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candle
import llama_cpp
import magnetar
import ollama
import strata
from common import summarize


def args(**kwargs):
    defaults = dict(device="cpu", model_path=None, binary=None, model="test", url=None,
                    tokens=16, runs=2)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class AdapterTests(unittest.TestCase):
    def test_summarize_ignores_missing_values_and_booleans(self):
        result = summarize([{"rate": 2.0, "other": None, "flag": True}, {"rate": 4.0}])
        self.assertEqual(result, {"rate": {"median": 3, "min": 2, "max": 4, "count": 2}})

    def test_candle_processes_and_gpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            config = args(model_path=str(model), binary="qwen")
            with patch.object(candle, "timed_command", return_value=("16 tokens generated (12.50 token/s)", "", 3.0)) as command:
                result = candle.run(config, Path(tmp) / "raw")
            self.assertEqual(command.call_count, 3)
            self.assertEqual(result["samples"][0]["generated_tokens"], 16)
            self.assertIn("--cpu", command.call_args.args[0])
            with patch.object(candle, "timed_command", return_value=("Running on CPU", "", 3.0)):
                with self.assertRaisesRegex(RuntimeError, "fell back"):
                    candle.run(args(device="gpu", model_path=str(model), binary="qwen"), Path(tmp) / "gpu")

    def test_llama_uses_decode_not_prompt_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "test.gguf"
            model.write_bytes(b"weights")
            log = ("llama_perf_context_print: prompt eval time = 50.00 ms / 5 tokens (100.00 tokens per second)\n"
                   "llama_perf_context_print: eval time = 200.00 ms / 10 runs (50.00 tokens per second)\n")
            with patch.object(llama_cpp, "timed_command", return_value=("", log, 5.0)):
                result = llama_cpp.run(args(binary="llama-completion", model_path=str(model)), Path(tmp) / "raw")
            self.assertEqual(result["samples"][0]["decode_tokens_per_second"], 50.0)
            self.assertIn("-no-cnv", result["command"])
            with patch.object(llama_cpp, "timed_command", return_value=("", log.splitlines()[0], 5.0)):
                with self.assertRaisesRegex(RuntimeError, "did not report eval"):
                    llama_cpp.run(args(binary="llama-completion", model_path=str(model)), Path(tmp) / "missing")
            with patch.object(llama_cpp, "timed_command", return_value=("", log + "offloaded 0/25 layers", 5.0)):
                with self.assertRaisesRegex(RuntimeError, "zero GPU layers"):
                    llama_cpp.run(args(device="gpu", binary="llama-completion", model_path=str(model)), Path(tmp) / "gpu")

    def test_ollama_checks_placement_and_real_token_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fake(url, body=None, **kwargs):
                if url.endswith("/api/ps"):
                    return {"models": [{"name": "test", "size_vram": 0}]}
                return {"done": True, "model": "test", "eval_count": 4, "eval_duration": 2_000_000_000}
            with patch.object(ollama, "http_json", side_effect=fake):
                result = ollama.run(args(), Path(tmp) / "cpu")
                self.assertEqual(result["samples"][0]["decode_tokens_per_second"], 2)
                with self.assertRaisesRegex(RuntimeError, "placement"):
                    ollama.run(args(device="gpu"), Path(tmp) / "gpu")

    def test_strata_uses_chat_api_and_requires_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 3}}
            with patch.object(strata, "http_json", return_value=response) as http:
                result = strata.run(args(url="http://localhost:8096"), Path(tmp) / "raw")
            self.assertEqual(len(result["samples"]), 2)
            self.assertTrue(http.call_args.args[0].endswith("/v1/chat/completions"))
            response["usage"] = {}
            with patch.object(strata, "http_json", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "completion_tokens"):
                    strata.run(args(url="http://localhost:8096"), Path(tmp) / "missing")

    def test_magnetar_requires_cuda_and_reads_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fake(command, cwd, env):
                self.assertEqual(env["MAGNETAR_QUALIFICATION_REQUIRE_CUDA"], "1")
                report = {"manifest": {"provider": {"name": "cuda"}, "checkpoint": {}, "dtype": "f32"},
                          "samples": [{"phase": "warmup", "metrics": {}},
                                      {"phase": "measured", "metrics": {"generated_tokens": 4}}]}
                Path(env["MAGNETAR_QUALIFICATION_REPORT_DIR"], "campaign.json").write_text(json.dumps(report))
                return "out", "err", 3.0
            with patch.object(magnetar, "timed_command", side_effect=fake):
                result = magnetar.run(args(device="gpu", model_path=tmp, runs=1), root / "raw")
            self.assertEqual(result["samples"], [{"generated_tokens": 4}])


if __name__ == "__main__":
    unittest.main()
