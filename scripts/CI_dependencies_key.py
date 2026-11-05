#!/usr/bin/env python3
"""Print the key of a configured dependency build (prebuilt_dependencies in scripts/CI_helpers.sh).

The key is a hash of what decides the installed dependencies (sources in third_party/<name>-src):
their source commits and local changes, the compile commands of their sources, the headers
generated for them (configuration options) and their generated CMake files (exported targets
with their usage requirements, install rules). It changes whenever the dependencies would be
built or installed differently, wherever in the CMake code the change comes from. Text on stdin
(image, loaded modules, the code of prebuilt_dependencies) is added to the key.

Usage: CI_dependencies_key.py <dependency build dir> < extra
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

build = Path(sys.argv[1]).absolute()
project = Path.cwd()
key = [sys.stdin.read()]


def normalize(text):
    return text.replace(str(project), "@PROJECT@")


entries = json.loads((build / "compile_commands.json").read_text())
key += sorted(normalize(e["command"]) for e in entries
              if Path(e["file"]).is_relative_to(project / "third_party"))

for source in sorted((project / "third_party").glob("*-src")):
    for args in (["rev-parse", "HEAD"], ["diff", "HEAD"]):
        key.append(subprocess.run(["git", "-C", source, *args], capture_output=True,
                                  text=True, check=True).stdout)

# Interface options that gempic sets on a dependency's target only show up in its exported targets
for pattern in ("*.[hH]*", "*.cmake"):
    for file in sorted(build.glob(f"_deps/*-build/**/{pattern}")):
        key += [str(file.relative_to(build)), normalize(file.read_text(errors="replace"))]

print(hashlib.sha256(json.dumps(key).encode()).hexdigest())
