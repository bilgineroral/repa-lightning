import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # repo root

# filled by the tests: (test id, quantity, max abs diff, tolerance used)
DIFFS = []

@pytest.fixture(autouse=True)
def no_tf32():
    """ full fp32 matmuls and convolutions in every test, restored afterwards. cuDNN allows TF32 by default
    (torch.backends.cudnn.allow_tf32 = True); on an RTX 6000 Ada it picked a TF32 algorithm for the patch-embedding
    Conv2d, so fp32 tests comparing it with the full-fp32 matmul of kern_patch_linear failed at ~3e-4 (TF32 precision).
    Setting both flags per test also removes the order dependence of tests that set them without restoring them """
    import torch
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved

def pytest_terminal_summary(terminalreporter):
    if not DIFFS:
        return
    terminalreporter.section("max differences vs. baseline")
    terminalreporter.write_line(f"{'test':58s} {'quantity':24s} {'max abs diff':>13s} {'tolerance':>11s}")
    for test, what, diff, tol in DIFFS:
        terminalreporter.write_line(f"{test:58s} {what:24s} {diff:13.3e} {tol:>11s}")
