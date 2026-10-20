"""Offline tests of actual patched server AST methods; no imports of the runtime, HTTP, or GPU work.

These tests do NOT establish native numerical correctness, real concurrency, or head compatibility.
"""
import argparse
import ast
import contextlib
import hmac
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import types
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
args = parser.parse_args()
tree = ast.parse((args.source / "serve/server.py").read_text(encoding="utf-8"))
classes = {"StrataEngine": {"cooperative_readout", "generate_batched", "_done_finish", "pick_slot"},
           "Service": {"latent_lease", "read_lease", "write_lease"}}
nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
for name, methods in classes.items():
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected} == methods
    nodes.append(ast.ClassDef(name=name, bases=[], keywords=[], decorator_list=[], body=selected))
helpers = {"readout_value", "key_list", "api_key_of", "key_matches"}
nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_handler")
handler = next(n for n in ast.walk(factory) if isinstance(n, ast.ClassDef))
methods = {"_post", "_authorized", "_latent"}
selected = [n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name in methods]
assert {n.name for n in selected} == methods
nodes.append(ast.ClassDef(name="Handler", bases=[], keywords=[], decorator_list=[], body=selected))
env = dict(contextlib=contextlib, threading=threading, queue=queue, time=time, os=os,
           tempfile=tempfile, hmac=hmac, json=json, btrace=lambda *a: None,
           EngineDied=RuntimeError, EOS_IDS=set())
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
             "<actual-patched-server-methods>", "exec"), env)
Engine, Service, Handler = env["StrataEngine"], env["Service"], env["Handler"]


def engine():
    e = Engine()
    e.batch = 4
    e.info = {"jev_slot_readout": 1, "batch_groups": 1}
    return e


def sampling():
    return {"latent_parallel": True, "latent_fast": True, "latent_path": "/tmp/test.bin",
            "latent_taps": "15,23,29,35,41,-1", "latent_spans": "0-10", "strata_checkpoint": False}


def service():
    s = Service()
    s.lease_cv = threading.Condition()
    s.lease_readers = 0
    s.lease_writer = False
    s.lease_waiting = 0
    s.latent_serial = threading.Lock()
    return s


class Admission(unittest.TestCase):
    def test_requires_native_capability(self):
        e = engine()
        self.assertTrue(e.cooperative_readout(sampling()))
        e.info.pop("jev_slot_readout")
        self.assertFalse(e.cooperative_readout(sampling()))

    def test_unsupported_groups_or_nonbatch_keep_exclusive(self):
        e = engine()
        e.info["batch_groups"] = 3
        self.assertFalse(e.cooperative_readout(sampling()))
        e.info["batch_groups"] = 1
        e.batch = 0
        self.assertFalse(e.cooperative_readout(sampling()))

    def test_two_groups_supported(self):
        e = engine()
        e.info["batch_groups"] = 2
        self.assertTrue(e.cooperative_readout(sampling()))

    def test_legacy_and_optout_keep_exclusive(self):
        for field, value in [("latent_parallel", False), ("latent_fast", False), ("latent_spans", None),
                             ("latent_taps", None), ("latent_kinds", "last")]:
            with self.subTest(field=field):
                p = sampling()
                p[field] = value
                self.assertFalse(engine().cooperative_readout(p))

    def test_busy_chat_slots_never_route_readout_to_bgen(self):
        e = engine()
        e.gen = 1
        e.progress = None
        e.alive = lambda: True
        e.sampling_keys = lambda s: " lat=/tmp/test.bin"
        e.slot_cv = threading.Condition()
        e.slot_busy = [True] * 4
        e.waiting = 2
        e.ctl = threading.Lock()
        e.last = {}
        e.sent = []
        e._send = e.sent.append
        e.YIELDS_MAX = 2
        e.slot_claims = set()
        e._shorter_waiting = lambda n: True
        e.pick_slot = lambda p: 0

        def take(*args, **kwargs):
            e.ctl.acquire()
            if False:
                yield
            return True

        def control(cancel, on_token, stop_when=None, on_quiet=None):
            self.assertIsNone(stop_when)
            self.assertIsNotNone(on_quiet)
            on_quiet()                 # upstream quiet-period yielding must not admit this GEN to BGEN
            yield None
            on_token(123)
            yield False
            e.last = {"finish": "length"}

        e._take_control, e._control = take, control
        self.assertEqual(list(e.generate_batched([1, 2, 3], 1, sampling(), threading.Event())), [None, 123])
        self.assertEqual(len(e.sent), 1)
        self.assertTrue(e.sent[0].startswith("GEN 1"))
        self.assertFalse(e.ctl.locked())
        self.assertEqual(e.slot_busy, [True] * 4)
        self.assertEqual(e.slot_claims, set())

    def test_cooperative_enters_while_chat_read_lease_held(self):
        s, entered = service(), threading.Event()

        def run():
            with s.latent_lease(True):
                entered.set()

        with s.read_lease():
            t = threading.Thread(target=run)
            t.start()
            self.assertTrue(entered.wait(1))
            t.join(1)
            self.assertFalse(t.is_alive())
        self.assertEqual(s.lease_readers, 0)

    def test_two_readouts_remain_serial(self):
        s, entered, started = service(), threading.Event(), threading.Event()

        def run():
            started.set()
            with s.latent_lease(True):
                entered.set()

        with s.latent_lease(True):
            t = threading.Thread(target=run)
            t.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(entered.wait(.05))
        self.assertTrue(entered.wait(1))
        t.join(1)
        self.assertEqual(s.lease_readers, 0)

    def test_legacy_excludes_chat(self):
        s, entered, release = service(), threading.Event(), threading.Event()

        def run():
            with s.latent_lease(False):
                entered.set()
                release.wait(2)

        with s.read_lease():
            t = threading.Thread(target=run)
            t.start()
            self.assertFalse(entered.wait(.05))
        self.assertTrue(entered.wait(1))
        self.assertTrue(s.lease_writer)
        release.set()
        t.join(1)
        self.assertFalse(s.lease_writer)

    def test_exception_releases_lease_and_serial_lock(self):
        s = service()
        with self.assertRaises(ValueError):
            with s.latent_lease(True):
                raise ValueError("client disconnected")
        self.assertEqual(s.lease_readers, 0)
        self.assertFalse(s.latent_serial.locked())


class GuardsAndAuth(unittest.TestCase):
    def test_command_control_characters_rejected(self):
        for v in ["x\nGEN", "x\rGEN", "x\tGEN", "x\vGEN", "x\fGEN", "x\x00GEN", "x GEN", "x\u2003GEN", "", None]:
            with self.subTest(value=repr(v)), self.assertRaises(ValueError):
                env["readout_value"]("latent_taps", v)
        self.assertEqual(env["readout_value"]("latent_taps", "15,23,-1"), "15,23,-1")

    def test_output_path_confined_to_temporary_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "lat.bin")
            self.assertEqual(env["readout_value"]("latent_path", p, path_of=True), p)
        with self.assertRaises(ValueError):
            env["readout_value"]("latent_path", str(Path(tempfile.gettempdir()).parent / "outside.bin"), path_of=True)

    def test_auth_router_20_cases(self):
        for key in ("", "test-shared-api-key"):
            env["svc"] = types.SimpleNamespace(api_key=env["api_key_of"](key),
                                               begin_request=lambda *a: None, drop_embeddings=lambda: None)
            for path in ("/v1/chat/completions", "/v1/latent"):
                for headers, expected in [({}, 401 if key else 200),
                                          ({"Authorization": "Bearer wrong"}, 401 if key else 200),
                                          ({"Authorization": "Bearer test-shared-api-key"}, 200),
                                          ({"x-api-key": "test-shared-api-key"}, 200),
                                          ({"X-Jev-Diagnostic-Token": "unrelated"}, 401 if key else 200)]:
                    obj, replies = Handler(), []
                    obj.path, obj.headers = path, headers
                    obj.watch_done, obj.record = None, None
                    obj._json = lambda status, body, **kw: replies.append((status, body))
                    obj._foreign_page = lambda: False
                    obj._body = lambda: b"{}"
                    obj._latent = lambda req: replies.append((200, {"route": "latent"}))
                    obj._openai = lambda req: replies.append((200, {"route": "chat"}))
                    obj._post()
                    self.assertEqual(len(replies), 1)
                    self.assertEqual(replies[0][0], expected, (path, headers))

    def test_actual_latent_handler_rejects_invalid_inputs(self):
        env["svc"] = types.SimpleNamespace(load=lambda: None, latent_n_layers=48,
                                          engine=types.SimpleNamespace(info={}))
        for changes in [{"latent_path": "caller.bin"}, {"latent_ids": [True]}, {"latent_ids": [-1]},
                        {"latent_ids": []}, {"latent_taps": "48"}, {"latent_taps": "47,-1"},
                        {"latent_taps": "15,15"}, {"latent_spans": [[0, 4]]},
                        {"latent_spans": [[1, 1]]}, {"latent_spans": [[0, 1]] * 17},
                        {"latent_pool": "last,last"}, {"latent_pool": "median"},
                        {"latent_taps": "15\nGEN"}]:
            req = {"latent_ids": [1, 2, 3], "latent_taps": "15", "latent_spans": [[0, 3]],
                   "latent_pool": "last,mean", **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Handler()._latent(req)


class SlotRetention(unittest.TestCase):
    def make(self):
        e = engine()
        e.slot_order = [0, 1, 2, 3]
        e.slot_busy = [False] * 4
        e.slot_held = [[], [], [], []]
        e.slot_groups, e.slot_gs, e.SLOT_PREFIX_MIN = 2, 2, 512
        return e

    def test_tool_tail_rewrite_keeps_prefix_owner(self):
        e = self.make()
        e.slot_held[2] = list(range(600)) + [9000]
        self.assertEqual(e.pick_slot(list(range(600)) + [9001]), 2)

    def test_short_overlap_does_not_override_balancing(self):
        e = self.make()
        e.slot_held[2] = list(range(511))
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot(list(range(511)) + [999]), 0)

    def test_busy_prefix_owner_never_reassigned(self):
        e = self.make()
        e.slot_held[2], e.slot_busy[2] = list(range(600)), True
        self.assertNotEqual(e.pick_slot(list(range(600))), 2)

    def test_full_slots_return_none(self):
        e = self.make()
        e.slot_busy = [True] * 4
        self.assertIsNone(e.pick_slot([1, 2, 3]))


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (Admission, GuardsAndAuth, SlotRetention))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"passed": result.wasSuccessful(), "tests": result.testsRun,
                      "scope": "actual Python AST methods; mocked engine I/O; no native/live/numeric acceptance"}))
    raise SystemExit(not result.wasSuccessful())
