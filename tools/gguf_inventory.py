#!/usr/bin/env python3
"""Prints what a GGUF file declares, from its header alone.

The header (metadata, then one record per tensor) is read from a local file or
by an HTTP range request, so a 40 GB export is inventoried by downloading the
first few tens of MiB:

    tools/gguf_inventory.py FILE_OR_URL [--head-mib 64] [--metadata] [--tensors]

By default it prints the metadata keys (without the tokenizer's tables) and the
tensors grouped by name pattern, with their types and shapes, and the bytes
each type takes. `--metadata` and `--tensors` print only that part. Standard
library only.
"""
import argparse
import collections
import re
import struct
import sys
import urllib.request

# ggml type id -> (name, values per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    # Strata's own type, not in llama.cpp: 64 values in 18 bytes, an f16 scale d
    # and two bits per value (value j in byte j / 4 at bit 2 * (j % 4)),
    # meaning (code - 1) * d.
    42: ("Q2_0", 64, 18),
}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}


class Header:
    def __init__(self, data):
        self.b = data
        self.pos = 0

    def take(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.pos + size > len(self.b):
            raise EOFError("the header is longer than the bytes read: raise --head-mib")
        (value,) = struct.unpack_from("<" + fmt, self.b, self.pos)
        self.pos += size
        return value

    def string(self):
        n = self.take("Q")
        if self.pos + n > len(self.b):
            raise EOFError("the header is longer than the bytes read: raise --head-mib")
        text = self.b[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return text

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 7:
            return bool(self.take("B"))
        if kind == 9:
            element = self.take("I")
            count = self.take("Q")
            if element == 8:
                for _ in range(count):
                    length = self.take("Q")
                    self.pos += length
                return ("strings", count)
            width = struct.calcsize("<" + SCALARS[element])
            if count <= 64:
                return [self.take(SCALARS[element]) for _ in range(count)]
            self.pos += count * width
            return ("array", count)
        return self.take(SCALARS[kind])

    def parse(self):
        if self.b[:4] != b"GGUF":
            raise ValueError("not a GGUF file")
        self.pos = 4
        version = self.take("I")
        tensor_count = self.take("Q")
        kv_count = self.take("Q")
        metadata = collections.OrderedDict()
        for _ in range(kv_count):
            key = self.string()
            metadata[key] = self.value(self.take("I"))
        tensors = []
        for _ in range(tensor_count):
            name = self.string()
            dims = [self.take("Q") for _ in range(self.take("I"))]
            kind = self.take("I")
            tensors.append((name, dims, kind, self.take("Q")))
        return version, metadata, tensors


def read_bytes(source, head_mib):
    count = head_mib * 1024 * 1024
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"Range": f"bytes=0-{count - 1}"})
        with urllib.request.urlopen(request) as response:
            return response.read()
    with open(source, "rb") as file:
        return file.read(count)


def tensor_bytes(dims, kind):
    name, values, size = TYPES[kind]
    elements = 1
    for d in dims:
        elements *= d
    return elements // values * size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--head-mib", type=int, default=64)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--tensors", action="store_true")
    args = parser.parse_args()
    version, metadata, tensors = Header(read_bytes(args.source, args.head_mib)).parse()
    both = not (args.metadata or args.tensors)
    print(f"GGUF version {version}: {len(metadata)} metadata keys, {len(tensors)} tensors")
    if args.metadata or both:
        for key, value in metadata.items():
            if key.startswith("tokenizer.ggml.") and not isinstance(value, (int, bool, float)) and not isinstance(value, str):
                print(f"{key} = {value}")
            elif isinstance(value, str) and len(value) > 100:
                print(f"{key} = {value[:100]!r}...")
            else:
                print(f"{key} = {value}")
    if args.tensors or both:
        totals = collections.Counter()
        groups = collections.OrderedDict()
        for name, dims, kind, _ in tensors:
            label = TYPES.get(kind, (f"type {kind}",))[0]
            if kind in TYPES:
                totals[label] += tensor_bytes(dims, kind)
            pattern = re.sub(r"\.\d+\.", ".N.", name)
            groups.setdefault(pattern, []).append((tuple(dims), label))
        for pattern, items in groups.items():
            types = sorted({t for _, t in items})
            shapes = sorted({s for s, _ in items})
            more = " ..." if len(shapes) > 3 else ""
            print(f"{pattern}  x{len(items)}  {types}  {shapes[:3]}{more}")
        print("bytes by type (this file's tensors):")
        for label, size in totals.most_common():
            print(f"  {label:8s} {size / 1e9:8.2f} GB")


if __name__ == "__main__":
    try:
        main()
    except EOFError as error:
        sys.exit(str(error))
