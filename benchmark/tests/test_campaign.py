import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import campaign_report
import llama_server


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
