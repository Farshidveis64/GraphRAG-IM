#!/usr/bin/env python3
"""Minimal dependency-free test runner (pytest-style functions also work under pytest)."""

import importlib
import inspect
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_MODULES = [
    "tests.test_dataset",
    "tests.test_diffusion",
    "tests.test_retriever",
    "tests.test_selection",
    "tests.test_metrics",
    "tests.test_inference",
]


def run() -> int:
    passed, failed, failures = 0, 0, []
    for mod_name in TEST_MODULES:
        module = importlib.import_module(mod_name)
        tests = [(n, f) for n, f in inspect.getmembers(module, inspect.isfunction)
                 if n.startswith("test_") and f.__module__ == mod_name]
        print(f"\n{mod_name}  ({len(tests)} tests)")
        for name, func in tests:
            try:
                func()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}")
                failures.append((mod_name, name, traceback.format_exc()))
                failed += 1
    print("\n" + "=" * 60)
    print(f"RESULT: {passed} passed, {failed} failed")
    for mod_name, name, tb in failures:
        print(f"\n{mod_name}.{name}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
