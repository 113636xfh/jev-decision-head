#!/usr/bin/env python3
"""Fast no-compiler regression/negative tests for release build tooling.

The ELF fixture contains synthetic CUDA payload bytes (not executable device code).
Tests prove fail-closed file/section accounting, NOT CUDA numerical correctness.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from compare_binary import inspect_elf
import verify

HERE = Path(__file__).resolve().parent


def fixture(path):
    names = b"\0.text\0.rodata\0.nv_fatbin\0.data\0.shstrtab\0"
    data = bytearray(256 + 6 * 64)
    data[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 62, 1, 128, 0, 256, 0, 64, 0, 0, 64, 6, 5)
    # RIP-relative load, deliberately susceptible to the retired normalizer's overreach.
    data[128:136] = b"\x48\x8b\x05\x01\x00\x00\x00\xc3"
    data[144:152] = b"CONSTANT"
    data[160:168] = b"FATBIN00"
    data[176:184] = b"DATA0000"
    data[192:192 + len(names)] = names
    for i, (name, flags, offset, size) in enumerate([
        (".text", 6, 128, 8), (".rodata", 2, 144, 8), (".nv_fatbin", 2, 160, 8),
        (".data", 3, 176, 8), (".shstrtab", 0, 192, len(names))], 1):
        struct.pack_into("<IIQQQQIIQQ", data, 256 + i * 64,
                         names.index(name.encode()), 3 if i == 5 else 1, flags, offset, offset, size, 0, 0, 1, 0)
    path.write_bytes(data)


class Tools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        fixture(self.a)
        self.b.write_bytes(self.a.read_bytes())

    def run_compare(self, expected, *extra, env=None):
        r = subprocess.run([sys.executable, str(HERE / "compare_binary.py"), str(self.a), str(self.b), *map(str, extra)],
                           capture_output=True, text=True, env=env, timeout=5)
        self.assertEqual(r.returncode, expected, r.stdout + r.stderr)
        return json.loads(r.stdout if r.stdout else r.stderr)

    def mutate(self, offset):
        data = bytearray(self.b.read_bytes())
        data[offset] ^= 1
        self.b.write_bytes(data)

    def test_identical(self):
        self.assertTrue(self.run_compare(0)["byte_identical"])

    def test_instruction_change(self):
        self.mutate(128)
        self.assertIn(".text", self.run_compare(1)["differing_sections"])

    def test_rip_displacement_not_erased(self):
        self.mutate(131)
        self.run_compare(1)

    def test_constant_only_change(self):
        self.mutate(144)
        self.assertIn(".rodata", self.run_compare(1)["differing_sections"])

    def test_cuda_fatbin_only_change(self):
        self.mutate(160)
        report = self.run_compare(1)
        self.assertEqual(report["differing_sections"], [".nv_fatbin"])
        self.assertEqual(report["left"]["sections"][1]["sha256"], report["right"]["sections"][1]["sha256"])

    def test_data_only_change(self):
        self.mutate(176)
        self.run_compare(1)

    def test_padding_only_change(self):
        self.mutate(240)
        report = self.run_compare(1)
        self.assertFalse(report["byte_identical"])
        self.assertEqual(report["differing_sections"], [])

    def test_missing_input(self):
        self.b.unlink()
        self.run_compare(2)

    def test_empty_input(self):
        self.b.write_bytes(b"")
        self.run_compare(2)

    def test_non_elf(self):
        self.b.write_bytes(b"not an ELF" * 10)
        self.run_compare(2)

    def test_truncated_elf(self):
        self.b.write_bytes(self.b.read_bytes()[:-1])
        self.run_compare(2)

    def test_no_text_section(self):
        data = self.b.read_bytes().replace(b".text\0", b".xxxx\0")
        self.b.write_bytes(data)
        self.run_compare(2)

    def test_objdump_not_a_dependency(self):
        # No external inspector can fail and be mistaken for an empty successful stream.
        # With PATH empty, byte identity still works; changed CUDA bytes still fail.
        env = {**os.environ, "PATH": ""}
        self.run_compare(0, env=env)
        self.mutate(160)
        self.run_compare(1, env=env)

    def test_report_never_overwritten(self):
        report = self.root / "report.json"
        self.run_compare(0, "--report", report)
        original = report.read_bytes()
        self.run_compare(2, "--report", report)
        self.assertEqual(report.read_bytes(), original)

    def test_report_creation_failure(self):
        self.run_compare(2, "--report", self.root / "missing-parent" / "report.json")

    def test_error_report(self):
        self.b.unlink()
        report = self.root / "report.json"
        self.run_compare(2, "--report", report)
        self.assertEqual(json.loads(report.read_text())["status"], "ERROR")

    def test_recipe_complete_and_shared(self):
        manifest = json.loads((HERE / "manifest.json").read_text())
        args = verify.cuda_configure_command(manifest, "source", "build", "ggml", "nvcc", "70")
        for key in ("AVX", "AVX2", "FMA", "F16C", "BMI2", "SSE42", "AVX512", "AVX512_BF16",
                    "AVX512_VBMI", "AVX512_VNNI", "AVX_VNNI", "AMX_TILE", "AMX_INT8", "AMX_BF16"):
            self.assertIn("-DGGML_" + key + "=OFF", args)
        self.assertIn("-DGGML_NATIVE=ON", args)
        self.assertIn("-DSTRATA_MMQ_KQUANTS=OFF", args)
        shell = (HERE / "reproduce-build.sh").read_text()
        self.assertIn('"$HERE/verify.py"', shell)
        self.assertNotIn("cmake -S", shell)
        self.assertNotIn("objdump", shell)

    def test_ggml_extra_or_changed_file_fails_digest(self):
        tree = self.root / "ggml"
        tree.mkdir()
        (tree / "a").write_bytes(b"source")
        before = verify.ggml_digest(tree)
        (tree / "extra").write_bytes(b"injected")
        self.assertNotEqual(before, verify.ggml_digest(tree))

    def test_ggml_digest_uses_posix_string_order(self):
        import hashlib
        tree = self.root / "ggml"
        (tree / "a").mkdir(parents=True)
        (tree / "a" / "b").write_bytes(b"nested")
        (tree / "a.c").write_bytes(b"sibling")
        rows = "a.c\0" + hashlib.sha256(b"sibling").hexdigest() + "\n"
        rows += "a/b\0" + hashlib.sha256(b"nested").hexdigest() + "\n"
        self.assertEqual(verify.ggml_digest(tree), hashlib.sha256(rows.encode()).hexdigest())

    def test_verifier_rejects_compare_without_build(self):
        r = subprocess.run([sys.executable, str(HERE / "verify.py"), "--work-dir", str(self.root / "work"),
                            "--compare-with", str(self.a)], capture_output=True, timeout=5)
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.root / "work").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
