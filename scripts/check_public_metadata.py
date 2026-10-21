#!/usr/bin/env python3
"""Check tracked text for private machine identity; never print matched values.

This complements a credential scanner. It is not a guarantee that no private
information exists, and deliberately skips untracked local runtime artifacts.
"""
import argparse
import re
import subprocess
from pathlib import Path


RULES = {
    "personal absolute path": re.compile(r"/(?:home|Users|mnt/data)/[A-Za-z0-9_.-]+"),
    "GPU hardware UUID": re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"),
    "session UUID": re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b"),
    "SSH endpoint with address": re.compile(r"[\w.-]+@(?:\d{1,3}\.){3}\d{1,3}"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Inspect a committed tree instead of the working tree")
    args = parser.parse_args()
    if args.revision:
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", args.revision]).decode().split("\0")
    else:
        paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    findings = []
    for name in filter(None, paths):
        if name.startswith(("runtime/", "skills/")):
            findings.append((name, 0, "private orchestration directory"))
        try:
            raw = (subprocess.check_output(["git", "show", f"{args.revision}:{name}"])
                   if args.revision else Path(name).read_bytes())
            text = raw.decode("utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            continue
        for line, value in enumerate(text.splitlines(), 1):
            for rule, pattern in RULES.items():
                if pattern.search(value):
                    findings.append((name, line, rule))
    for name, line, rule in findings:
        print(f"{name}:{line}: {rule}")
    print(f"Public metadata check: {len(findings)} finding(s). Matched values are not displayed.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
