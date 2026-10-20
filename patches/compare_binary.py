#!/usr/bin/env python3
"""Strict, fail-closed ELF comparison. No disassembly or normalization is an acceptance criterion.

Exit 0: entire files byte-identical; 1: valid files differ; 2: inspection/tool/report error.
Section hashes are diagnostics, including CUDA fatbins and constants, NOT equivalence proofs.
Requires Python >=3.9; never loads or executes the inspected binary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect_elf(path, require_executable=True):
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError(f"{path}: requires a nonempty ELF64 little-endian file")
    fields = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    kind, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, shstr = fields
    if machine != 62 or version != 1 or ehsize != 64 or kind not in (1, 2, 3):
        raise ValueError(f"{path}: unsupported ELF identity/header")
    if require_executable and kind not in (2, 3):
        raise ValueError(f"{path}: not an executable/shared ELF")

    def bounded(offset, size):
        if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
            raise ValueError(f"{path}: out-of-bounds ELF range {offset}+{size}")
        return data[offset:offset + size]

    if not shnum or shsize != 64 or shstr >= shnum:
        raise ValueError(f"{path}: missing/unsupported section table (extended counts are rejected)")
    bounded(shoff, shsize * shnum)
    if phnum:
        if phsize != 56:
            raise ValueError(f"{path}: invalid program header size")
        bounded(phoff, phsize * phnum)
        for i in range(phnum):
            p = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phsize)
            bounded(p[2], p[5])
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize) for i in range(shnum)]
    names_h = headers[shstr]
    if names_h[1] != 3:
        raise ValueError(f"{path}: section names are not a string table")
    names = bounded(names_h[4], names_h[5])
    rows = []
    for i, (name_i, typ, flg, addr, offset, size, link, info, align, entsize) in enumerate(headers):
        if name_i >= len(names) or names.find(b"\0", name_i) < 0:
            raise ValueError(f"{path}: invalid section name")
        name = names[name_i:names.index(b"\0", name_i)].decode("utf-8", errors="strict")
        # SHT_NULL may store extended-header metadata; SHT_NOBITS has no bytes on disk.
        section_hash = None if typ in (0, 8) else digest(bounded(offset, size))
        rows.append(dict(index=i, name=name, type=typ, flags=flg, address=addr,
                         offset=offset, size=size, sha256=section_hash))
    text = [r for r in rows if r["name"] == ".text" and r["size"] and r["flags"] & 4]
    if require_executable and len(text) != 1:
        raise ValueError(f"{path}: expected one nonempty executable .text section")
    return dict(path=str(path.resolve()), bytes=len(data), sha256=digest(data),
                elf_type=kind, machine=machine, sections=rows)


def compare(a, b):
    left, right = inspect_elf(a), inspect_elf(b)
    same = left["bytes"] == right["bytes"] and left["sha256"] == right["sha256"]
    # Hash equality is accompanied by a literal byte comparison, never used alone.
    if same:
        with Path(a).open("rb") as fa, Path(b).open("rb") as fb:
            while True:
                ca, cb = fa.read(1024 * 1024), fb.read(1024 * 1024)
                if ca != cb:
                    same = False
                    break
                if not ca:
                    break
    keys = sorted({r["name"] for r in left["sections"] + right["sections"]})
    changed = []
    for key in keys:
        la = [{k: r[k] for k in ("type", "flags", "size", "sha256")} for r in left["sections"] if r["name"] == key]
        rb = [{k: r[k] for k in ("type", "flags", "size", "sha256")} for r in right["sections"] if r["name"] == key]
        if la != rb:
            changed.append(key)
    return dict(schema="small-decider-strict-binary-comparison-v1",
                inspected_utc=datetime.now(timezone.utc).isoformat(),
                status="PASS" if same else "FAIL", criterion="entire_file_byte_identity",
                byte_identical=same, normalized=False, left=left, right=right,
                differing_sections=changed,
                boundary="Section matches are scoped byte facts only; differences never establish semantic equivalence. No runtime/numerical acceptance.")


def write_new_report(path, report):
    # Never replace an earlier audit report, even on a failed comparison.
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--report", type=Path, help="NEW JSON report path; never overwritten")
    args = p.parse_args(argv)
    try:
        report = compare(args.left, args.right)
        if args.report:
            write_new_report(args.report, report)
        print(json.dumps(report, indent=2))
        return 0 if report["byte_identical"] else 1
    except Exception as exc:
        report = dict(schema="small-decider-strict-binary-comparison-v1", status="ERROR",
                      byte_identical=False, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(report, indent=2), file=sys.stderr)
        if args.report and not args.report.exists():
            try:
                write_new_report(args.report, report)
            except OSError:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
