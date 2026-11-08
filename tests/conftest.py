import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # repo root

# filled by the tests: (test id, quantity, max abs diff, tolerance used)
DIFFS = []

def pytest_terminal_summary(terminalreporter):
    if not DIFFS:
        return
    terminalreporter.section("max differences vs. baseline")
    terminalreporter.write_line(f"{'test':58s} {'quantity':24s} {'max abs diff':>13s} {'tolerance':>11s}")
    for test, what, diff, tol in DIFFS:
        terminalreporter.write_line(f"{test:58s} {what:24s} {diff:13.3e} {tol:>11s}")
