"""Generate Laya text layouts using pinned upstream code and tokenizer.

Usage: python generate_laya_text_reference.py TOKENIZER.json RL_COMMON.py OUTPUT.json
The input hashes are checked before running the actual upstream functions.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizerFast

tokenizer_path, source_path, output_path = map(Path, sys.argv[1:])
assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e"
tree = ast.parse(source)
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"serialize_state", "render_options", "build_sequence"}]
assert len(functions) == 3
upstream = {"json": json, "Dict": Dict, "List": List, "Optional": Optional}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), upstream)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_path), cls_token="[CLS]", sep_token="[SEP]",
    mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]",
)
cases = [
    {"state": "The sky is clear and dry.", "question": {
        "t": "choice", "ins": "Choose a forecast", "crit": {"sun": "clear", "rain": "wet", "snow": None}}},
    {"state": "Température: 21 °C [MASK]", "question": {
        "t": "score", "ins": "Rate comfort [MASK]", "crit": ["cold", "pleasant", "hot"]}},
    {"state": "It is raining.", "question": {
        "t": "noul", "ins": "The ground is wet", "crit": {"false": "dry", "true": "wet"}}},
]
for case in cases:
    if case["question"]["t"] == "choice":
        case["option_order"] = list(case["question"]["crit"])
    case["ids"], case["markers"] = upstream["build_sequence"](
        tokenizer, case["state"], case["question"], 512, 192,
    )
output_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
