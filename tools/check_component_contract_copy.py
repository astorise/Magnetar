#!/usr/bin/env python3
"""The Runtime's copy of the Model Component contract is the API repository's.

`magnetar-runtime/wit/model-component-graph.wit` is a copy of one released
version of the contract, which lives in
https://github.com/astorise/magnetar-component-api (`wit/<version>/`, immutable
once released: `openspec/changes/introduce-component-api`, task 3.1). This
check reads the version the Runtime's file declares and fails unless the file
is, byte for byte, the repository's file of that version at API_REV.

A new contract version is released in the API repository first; this check
then fails until API_REV is moved to a commit that has it and the Runtime's
file is updated to match, in one pull request.

    python3 tools/check_component_contract_copy.py
    python3 tools/check_component_contract_copy.py --api-file <path>   # offline
"""

import argparse
import hashlib
import pathlib
import re
import sys
import urllib.error
import urllib.request

# A commit of magnetar-component-api that holds the version the Runtime serves.
API_REV = "2383a7479ff496b147e27e294f3190d5209b39d4"
RUNTIME_FILE = pathlib.Path("magnetar-runtime/wit/model-component-graph.wit")
RAW = "https://raw.githubusercontent.com/astorise/magnetar-component-api/{rev}/wit/{version}/model-component-graph.wit"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api-file", help="compare with this local file instead of fetching it")
    args = parser.parse_args()

    runtime = RUNTIME_FILE.read_bytes()
    match = re.search(rb"^package magnetar:model-component-graph@(\d+\.\d+\.\d+);", runtime, re.M)
    if not match:
        print(f"{RUNTIME_FILE} declares no `package magnetar:model-component-graph@<version>;`")
        return 1
    version = match.group(1).decode()

    if args.api_file:
        api = pathlib.Path(args.api_file).read_bytes()
        source = args.api_file
    else:
        source = RAW.format(rev=API_REV, version=version)
        try:
            with urllib.request.urlopen(source, timeout=60) as response:
                api = response.read()
        except urllib.error.HTTPError as error:
            print(
                f"magnetar-component-api has no wit/{version}/ at {API_REV} ({error}).\n"
                "Release the version there first, then move API_REV in this script."
            )
            return 1

    if runtime == api:
        print(f"contract {version}: the Runtime's copy is the API repository's ({digest(api)[:16]})")
        return 0
    print(
        f"contract {version}: {RUNTIME_FILE} differs from the API repository's file\n"
        f"  Runtime {digest(runtime)}\n"
        f"  API     {digest(api)}  ({source})\n"
        "A released version never changes: if the Runtime's file was edited, the edit\n"
        "belongs in a new version of magnetar-component-api, released there first."
    )
    return 1


sys.exit(main())
