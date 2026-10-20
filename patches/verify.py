#!/usr/bin/env python3
"""Validate the pinned patch in a NEW sandbox, optionally compile CUDA; never run a model or install files.

Requires Python >= 3.9 and git; optional build requires CMake, a C++ compiler, CUDA and pinned ggml sources.
The original --source checkout must be clean and is never patched. Reports FAIL on any unsuccessful stage.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_binary import inspect_elf

HERE = Path(__file__).resolve().parent
COMMIT = "296b59d2f73c8f04a805df29c2fa97655adace78"
TREE = "063c7282837ecfe9fccad829ff7d7873400f6777"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ggml_digest(root):
    """Hash the exact cached ggml input tree, including any extra files (fail closed)."""
    rows = []
    root = Path(root)
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_file():
            rows.append(path.relative_to(root).as_posix() + "\0" + sha(path) + "\n")
    if not rows:
        raise ValueError("empty ggml source tree")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def cuda_configure_command(manifest, sandbox, build, ggml, compiler, arch):
    """The ONLY configure recipe, shared by verify.py and reproduce-build.sh."""
    recipe = manifest["build_recipe"]
    return ["cmake", "-S", str(sandbox), "-B", str(build)] + [
        "-D" + k + "=" + str(v) for k, v in recipe["cmake_definitions"].items()
    ] + ["-DCMAKE_CUDA_ARCHITECTURES=" + arch,
         "-DCMAKE_CUDA_COMPILER=" + compiler, "-DSTRATA_GGML_DIR=" + str(ggml)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, help="clean pinned checkout; omitted: clone upstream into the new work directory")
    p.add_argument("--upstream", default="https://github.com/Niko1221/Strata.git", help="clone URL/local mirror when --source is omitted")
    p.add_argument("--work-dir", type=Path, required=True, help="NEW/empty directory for sandbox, logs and report")
    p.add_argument("--build", choices=["none", "cuda"], default="none")
    p.add_argument("--ggml-dir", type=Path, help="cached llama.cpp tree containing ggml/; required for --build cuda")
    p.add_argument("--cuda-compiler", default="/usr/local/cuda-12.9/bin/nvcc")
    p.add_argument("--cuda-arch", default="70")
    p.add_argument("--jobs", type=int, default=1, help="compile parallelism (default: 1)")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    work = args.work_dir.resolve()
    source = args.source.resolve() if args.source else work / "upstream"
    if (args.source and (work == source or source in work.parents or work in source.parents)) or work == HERE or work in HERE.parents or HERE in work.parents:
        p.error("work-dir must be separate from the source and patch package")
    if work.exists() and any(work.iterdir()):
        p.error("work-dir is not empty; preserve existing evidence and choose a new directory")
    work.mkdir(parents=True, exist_ok=True)
    sandbox = work / "source"
    report = dict(schema="small-decider-patch-validation-v1", status="RUNNING", steps=[],
                  upstream_commit=COMMIT, upstream_tree=TREE,
                  started_utc=datetime.now(timezone.utc).isoformat(), build_requested=args.build,
                  build_status="NOT_RUN",
                  native_runtime_status="NOT_RUN", numeric_compatibility="NOT_MEASURED",
                  deployment_modified=False, gpu_inference=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}

    def redact(text):
        for old, new in [(str(source), "<upstream>"), (str(work), "<work>"), (str(HERE), "<patches>")]:
            text = text.replace(old, new).replace(old.replace("\\", "/"), new)
        if args.ggml_dir:
            text = text.replace(str(args.ggml_dir.resolve()), "<ggml>")
        return text

    def save():
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(name, argv):
        save()
        with (work / "commands.log").open("a", encoding="utf-8") as log:
            log.write("\n$ " + redact(json.dumps([str(a) for a in argv])) + "\n")
            log.flush()
            # Stream compiler output to disk, not RAM. A failed/unfinished build is never reported as PASS.
            with tempfile_log.open("wb") as raw:
                result = subprocess.run([str(a) for a in argv], env=env, stdout=raw, stderr=subprocess.STDOUT)
            text = tempfile_log.read_text(encoding="utf-8", errors="replace")
            log.write(redact(text))
        report["steps"].append(dict(name=name, returncode=result.returncode))
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed (rc={result.returncode}); see commands.log")
        return text.strip()

    tempfile_log = work / "last-command.log"
    try:
        manifest_path = HERE / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["upstream"]["commit"] != COMMIT or manifest["upstream"]["tree"] != TREE:
            raise ValueError("manifest has a different upstream identity")
        report["manifest_sha256"] = sha(manifest_path)
        patches = manifest["patches"]
        for row in patches + manifest["validation_tools"]:
            if Path(row["file"]).name != row["file"] or sha(HERE / row["file"]) != row["sha256"]:
                raise ValueError("patch/tool integrity mismatch: " + row["file"])
        if not args.source:
            run("clone_upstream", ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false",
                                   "clone", "--no-checkout", args.upstream, str(source)])
            run("pin_upstream", ["git", "-c", "core.autocrlf=false", "-C", str(source),
                                 "checkout", "--detach", COMMIT])
        basegit = ["git", "--no-optional-locks", "-c", "core.longpaths=true", "-C", str(source)]
        if run("source_commit", basegit + ["rev-parse", "HEAD"]) != COMMIT:
            raise ValueError("source HEAD is not the pinned upstream commit")
        if run("source_tree", basegit + ["rev-parse", "HEAD^{tree}"]) != TREE:
            raise ValueError("source tree is not the pinned upstream tree")
        if run("source_clean", basegit + ["status", "--porcelain", "--untracked-files=all"]):
            raise ValueError("source must be clean; no reset/clean is performed")
        run("clone", ["git", "-c", "core.longpaths=true", "clone", "--shared", "--no-checkout", str(source), str(sandbox)])
        git = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", "-C", str(sandbox)]
        run("checkout", git + ["checkout", "--detach", COMMIT])
        if run("sandbox_clean", git + ["status", "--porcelain", "--untracked-files=all"]):
            raise ValueError("new sandbox checkout is not clean")
        for row in manifest["files"]:
            if sha(sandbox / row["path"]) != row["base_sha256"]:
                raise ValueError("base file bytes differ: " + row["path"])
        for i, row in enumerate(patches):
            patch = HERE / row["file"]
            run(f"apply_check_{i}", git + ["apply", "--check", str(patch)])
            run(f"apply_{i}", git + ["apply", str(patch)])
        for row in manifest["files"]:
            if sha(sandbox / row["path"]) != row["patched_sha256"]:
                raise ValueError("patched file bytes differ: " + row["path"])
        changed = run("changed_paths", git + ["diff", "--name-only"]).splitlines()
        if sorted(changed) != sorted(row["path"] for row in manifest["files"]):
            raise ValueError("patch changed unexpected paths")
        run("diff_check", git + ["diff", "--check"])
        ast.parse((sandbox / "serve/server.py").read_text(encoding="utf-8"))
        report["python_syntax"] = "PASS"
        run("offline_contract", [sys.executable, str(HERE / "test_contract.py"), str(sandbox)])
        report["apply_status"] = "PASS"
        report["patched_files_sha256"] = {row["path"]: sha(sandbox / row["path"]) for row in manifest["files"]}
        if args.build == "cuda":
            report["build_status"] = "RUNNING"
            if not args.ggml_dir or not (args.ggml_dir / "ggml/CMakeLists.txt").is_file():
                raise ValueError("--ggml-dir must contain cached pinned ggml sources")
            ggml = args.ggml_dir.resolve()
            report["ggml_tree_sha256"] = ggml_digest(ggml / "ggml")
            if report["ggml_tree_sha256"] != manifest["build_recipe"]["ggml_tree_sha256"]:
                raise ValueError("ggml input-tree digest does not match the audited cached build inputs")
            build = work / "build"
            argv = cuda_configure_command(manifest, sandbox, build, ggml, args.cuda_compiler, args.cuda_arch)
            report["configure_argv"] = [redact(str(a)) for a in argv]
            report["parallel_jobs"] = args.jobs
            report["build_environment"] = {k: redact(os.environ.get(k, "")) for k in (
                "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CUDAFLAGS", "CUDACXX", "CUDAHOSTCXX", "SOURCE_DATE_EPOCH")}
            run("configure_cuda", argv)
            cache = {}
            for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
                if "=" in line and ":" in line.split("=", 1)[0] and not line.startswith(("#", "//")):
                    key, value = line.split("=", 1)
                    cache[key.split(":", 1)[0]] = value
            expected = {**manifest["build_recipe"]["cmake_definitions"],
                        "CMAKE_CUDA_ARCHITECTURES": args.cuda_arch, "CMAKE_CUDA_COMPILER": args.cuda_compiler,
                        "STRATA_GGML_DIR": str(ggml)}
            if any(cache.get(k) != str(v) for k, v in expected.items()):
                raise ValueError("configured CMake cache differs from the complete recipe")
            report["cmake_cache_sha256"] = sha(build / "CMakeCache.txt")
            run("build_cuda", ["cmake", "--build", str(build), "--target", "strata", "-j" + str(args.jobs)])
            binary = build / ("strata.exe" if os.name == "nt" else "strata")
            if os.name != "nt":
                inventory = inspect_elf(binary)
                if not any(r["name"] == ".nv_fatbin" and r["size"] for r in inventory["sections"]):
                    raise ValueError("CUDA build produced no nonempty .nv_fatbin section")
                report["built_elf_sections"] = inventory["sections"]
            if binary.stat().st_size == 0:
                raise ValueError("build produced an empty binary")
            report.update(build_status="PASS", binary_sha256=sha(binary), binary_bytes=binary.stat().st_size)
        report["status"] = "PASS"
        report["scope"] = ("pinned clean apply + offline Python tests + optional compile; "
                           "no model requests, runtime/numeric acceptance, installation or restart")
    except Exception as exc:
        report.update(status="FAIL", error=redact(f"{type(exc).__name__}: {exc}"))
        if report["build_status"] == "RUNNING":
            report["build_status"] = "FAIL"
    finally:
        save()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
