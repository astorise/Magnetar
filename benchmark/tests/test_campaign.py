import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import campaign_report
import llama_server
import qwen36_preflight
import strata_native


class FakeProcess:
    def __init__(self, command, stdout, stderr):
        stdout.write("offloaded 25/25 layers\n")
        stdout.flush()
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class CampaignTests(unittest.TestCase):
    def test_experimental_llama_records_the_exact_ids_used_by_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"same weights")
            ids = root / "prompt.json"
            requests = []
            args = SimpleNamespace(binary="server", model_path=str(model), device="gpu", tokens=2, runs=1)

            def http(url, body=None, timeout=3600):
                requests.append((url, body))
                if url.endswith("/health"):
                    return {"status": "ok"}
                if url.endswith("/tokenize"):
                    return {"tokens": [11, 22]}
                return {"content": "ok", "tokens": [33, 44],
                        "timings": {"prompt_n": 2, "predicted_n": 2, "predicted_per_second": 1}}

            with patch.dict(os.environ, {"BENCH_MODEL_PROFILE": "qwen3.6-35b-experimental", "BENCH_PROMPT_IDS_PATH": str(ids)}), \
                    patch.object(llama_server.subprocess, "Popen", side_effect=FakeProcess), \
                    patch.object(llama_server, "http_json", side_effect=http):
                result = llama_server.run(args, root / "raw")
            self.assertEqual(json.loads(ids.read_text()), [11, 22])
            self.assertEqual([body["prompt"] for url, body in requests if url.endswith("/completion")], [[11, 22], [11, 22]])
            self.assertIn("--cpu-moe", result["command"])
            self.assertEqual(result["samples"][0]["generated_token_ids"], [33, 44])

    def test_strata_native_uses_same_prompt_and_preserves_cache_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"same weights")
            ids = root / "prompt.json"
            ids.write_text("[11, 22]")
            args = SimpleNamespace(binary=str(root / "strata"), model_path=str(model), device="gpu", tokens=2, runs=1)

            class Process:
                def __init__(self, *args, **kwargs):
                    self.stdin = io.StringIO()
                    self.stdout = io.StringIO("INFO test=1\nREADY 512 stop\nT 33\nT 44\nDONE 2 2 1 2 length 0 0 0\n"
                                              "T 33\nT 44\nDONE 2 2 1 2 length 0 0 1\n")
                def wait(self, timeout=None):
                    return 0

            with patch.dict(os.environ, {"BENCH_PROMPT_IDS_PATH": str(ids), "CUDA_VISIBLE_DEVICES": "0"}), \
                    patch.object(strata_native.subprocess, "Popen", side_effect=Process):
                result = strata_native.run(args, root / "raw")
            self.assertEqual(result["samples"][0]["generated_token_ids"], [33, 44])
            self.assertEqual(result["samples"][0]["cached_prompt_tokens"], 1)
            self.assertEqual(result["samples"][0]["prompt_tokens"], 2)
            self.assertIn("--mtp", result["command"])

    def test_large_model_preflight_refuses_insufficient_memory(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(qwen36_preflight, "effective_memory", return_value=16_000_000_000):
            with self.assertRaisesRegex(RuntimeError, "32 GB RAM"):
                qwen36_preflight.main(tmp, Path(tmp) / "model.gguf")

    def test_experimental_summary_keeps_unknown_tokens_and_strata_cache_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-profile.txt").write_text("qwen3.6-35b-experimental", encoding="utf-8")
            (root / "status.tsv").write_text("strata_native\tsuccess\ncandle\tunavailable\n", encoding="utf-8")
            path = root / "strata_native" / "report.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"schema": "magnetar.benchmark/v1", "engine": "strata_native",
                "samples": [], "summary": {"cached_prompt_tokens": {"max": 3}}}), encoding="utf-8")
            text = campaign_report.render(root)
            self.assertIn("Experimental profile", text)
            self.assertIn("Strata reused up to 3 prompt tokens", text)
            self.assertIn("Prompt token counts are unavailable", text)
            self.assertNotIn("Token counts differ", text)

    def test_flash_next_summary_reports_the_single_engine_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-profile.txt").write_text("qwen3.8-flash-next-q2-experimental", encoding="utf-8")
            (root / "status.tsv").write_text("magnetar\tunavailable\nstrata\tsuccess\n", encoding="utf-8")
            text = campaign_report.render(root)
            self.assertIn("Strata only", text)
            self.assertIn("strata-engine.log", text)
            self.assertIn("Short outputs can contain only reasoning", text)
            self.assertIn("| magnetar | unavailable |", text)

    def test_llama_server_is_local_disables_cache_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "test.gguf"
            model.write_bytes(b"test")
            args = SimpleNamespace(binary="server", model_path=str(model), device="gpu", tokens=8, runs=2)
            process = []
            requests = []

            def spawn(*a, **kw):
                result = FakeProcess(*a, **kw)
                process.append(result)
                return result

            def http(url, body=None, timeout=3600):
                requests.append((url, body))
                if url.endswith("/health"):
                    return {"status": "ok"}
                return {"content": " Paris", "timings": {"prompt_n": 5, "predicted_n": 8, "predicted_per_second": 4.0}}

            with patch.object(llama_server.subprocess, "Popen", side_effect=spawn), patch.object(llama_server, "http_json", side_effect=http):
                result = llama_server.run(args, root / "raw")
            self.assertEqual(len(result["samples"]), 2)
            self.assertTrue(process[0].terminated)
            self.assertTrue(all(url.startswith("http://127.0.0.1:") for url, _ in requests))
            self.assertTrue(all(body["cache_prompt"] is False for _, body in requests[1:]))
            self.assertEqual(result["samples"][0]["prompt_tokens"], 5)
            self.assertIn("25/25", result["device_evidence"])
            self.assertIn("--verbose", result["command"])

    def test_summary_keeps_failure_and_warns_on_token_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.tsv").write_text("magnetar\tsuccess\nairllm\tfailed\nstrata\tunavailable\n", encoding="utf-8")
            for engine, count in (("magnetar", 5), ("llama_server", 6)):
                path = root / engine / "report.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"schema": "magnetar.benchmark/v1", "engine": engine,
                    "samples": [{"generated_token_ids": [count]}],
                    "summary": {"prompt_tokens": {"median": count}, "generated_tokens": {"median": 8},
                                "request_seconds": {"median": 2}, "request_tokens_per_second": {"median": 4}}}), encoding="utf-8")
            text = campaign_report.render(root)
            self.assertIn("| airllm | failed |", text)
            self.assertIn("| strata | unavailable |", text)
            self.assertIn("Token counts differ", text)
            self.assertIn("Generated token sequences differ", text)

    def test_llama_server_rejects_unverified_gpu_and_still_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "test.gguf"
            model.write_bytes(b"test")
            args = SimpleNamespace(binary="server", model_path=str(model), device="gpu", tokens=8, runs=1)
            process = FakeProcess([], io.StringIO(), None)
            with patch.object(llama_server.subprocess, "Popen", return_value=process), \
                    patch.object(llama_server, "http_json", return_value={"status": "ok"}):
                with self.assertRaisesRegex(RuntimeError, "did not confirm GPU"):
                    llama_server.run(args, root / "raw")
            self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
