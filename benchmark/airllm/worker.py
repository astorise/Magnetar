"""AirLLM worker; its process is isolated so all device memory is released afterwards."""
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from airllm import AutoModel

PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available to PyTorch")
    device = "cuda:0" if args.device == "gpu" else "cpu"
    model = AutoModel.from_pretrained(args.model_path, device=device, dtype=torch.float32)
    inputs = model.tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    samples = []
    for index in range(args.runs + 1):
        if args.device == "gpu":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False,
                                    use_cache=True, return_dict_in_generate=True)
        if args.device == "gpu":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output.sequences[0, input_ids.shape[1]:]
        count = int(generated.shape[0])
        if count < 1:
            raise RuntimeError("AirLLM generated no tokens")
        if index:
            samples.append({"prompt_tokens": int(input_ids.shape[1]), "generated_tokens": count,
                            "request_seconds": elapsed, "request_tokens_per_second": count / elapsed,
                            "generated_token_ids": generated.tolist(),
                            "text": model.tokenizer.decode(generated, skip_special_tokens=False)})
    result = {"device_evidence": {"device": str(model.device), "cuda_available": torch.cuda.is_available(),
                                  "gpu": torch.cuda.get_device_name(0) if args.device == "gpu" else None},
              "model": {"path": str(Path(args.model_path).resolve())}, "dtype": "float32",
              "runtime_versions": {"torch": torch.__version__, "transformers": version("transformers"),
                                   "airllm": version("airllm")}, "samples": samples}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
