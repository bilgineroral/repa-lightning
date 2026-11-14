""" Tier 3 runner logic (bench/kern_preflight.py, bench/kern_runner.py): CPU only, no GPU or subprocess needed. """
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import kern_preflight as kp # noqa: E402
import kern_runner as kr # noqa: E402

def test_select_env_by_driver():
    assert kp.select_env("580.65.06")[0] == "repa-kern"
    assert kp.select_env("617.14")[0] == "repa-kern"
    assert kp.select_env("579.99")[0] == "repa-kern-cu126"
    assert kp.select_env("525.60.13")[0] == "repa-kern-cu126"
    assert kp.select_env("470.82")[0] is None
    assert kp.select_env("n/a")[0] is None

def test_capability_warning():
    assert kp.capability_warning("8.9") is None
    assert "8.6" in kp.capability_warning("8.6")

def test_is_busy():
    assert kp.is_busy([], 1400, 1, 2048, 20) == [] # the idle WSL 3060: display memory only
    assert kp.is_busy(["GPU-x, 123, python, 3000 MiB"], 10, 0, 2048, 20)
    assert kp.is_busy([], 5000, 0, 2048, 20) # WSL: no process list, memory gives it away
    assert kp.is_busy([], 100, 95, 2048, 20)

def _ids(plan):
    return [s.id for s in plan]

def test_plan_order_and_dry_run():
    full = kr.build_plan(["S", "B"], dry_run=False)
    ids = _ids(full)
    assert len(ids) == len(set(ids))
    assert [i[:2] for i in ids] == sorted(i[:2] for i in ids) # the doc's order: 00 .. 06
    assert "02_compile_B_max-autotune" in ids and "03_fp8_S_rowwise_default" in ids
    dry = kr.build_plan(["S"], dry_run=True)
    hw = {"capability": [8, 9], "sms": 142} # even on an Ada GPU, a dry run skips the sm_89-only steps
    args = SimpleNamespace(dry_run=True, data_root=None)
    skipped = {s.id for s in dry if kr.unmet(s, hw, args, set())}
    assert all(i in skipped for i in _ids(dry) if i.startswith("03_fp8") or "max-autotune" in i)
    assert "02_compile_S_default" not in skipped and "02_compile_S_reduce-overhead" not in skipped
    assert "06_data_S_real" in skipped and "05_patch_linear_S" in skipped

def test_unmet_hardware_gates():
    s = kr.Step("x", "time", "S", requires=["fp8"])
    t = kr.Step("y", "time", "S", requires=["templates"])
    args = SimpleNamespace(dry_run=False, data_root=None)
    assert kr.unmet(s, {"capability": [8, 6], "sms": 28}, args, set())
    assert kr.unmet(t, {"capability": [8, 6], "sms": 28}, args, set())
    assert kr.unmet(s, {"capability": [8, 9], "sms": 142}, args, set()) is None
    assert kr.unmet(t, {"capability": [8, 9], "sms": 142}, args, set()) is None
    c = kr.Step("z", "time", "S", requires=["config:kern_patch_linear"])
    assert kr.unmet(c, {"capability": [8, 9], "sms": 142}, args, {"kern_patch_linear"}) is None

def _record(step, ms=100.0, status="ok"):
    return {"step": step.id, "status": status, "error": None if status == "ok" else "boom",
            "settings": {"mode": step.mode, "coord_descent": step.coord_descent, "fp8_recipe": step.fp8_recipe,
                         "flags": step.flags},
            "result": {"ms_per_step_median": ms}, "duration_s": 1.0}

def test_run_plan_resumes_and_retries_failures(tmp_path):
    plan = [kr.Step("01_a", "time", "S"), kr.Step("02_b", "time", "S"), kr.Step("03_c", "time", "S")]
    calls = []

    def execute_fail_b(step, s):
        calls.append(step.id)
        return _record(step, status="failed" if step.id == "02_b" else "ok")
    resolve = lambda step, results: {"mode": step.mode, "coord_descent": False, "fp8_recipe": "", "flags": "all"}
    st = kr.run_plan(plan, str(tmp_path), execute_fail_b, lambda s: None, resolve, lambda m: None)
    assert calls == ["01_a", "02_b", "03_c"] and st["02_b"][0] == "failed"
    assert (tmp_path / "02_b.failed.json").exists() and not (tmp_path / "02_b.json").exists()

    calls.clear()
    st = kr.run_plan(plan, str(tmp_path), lambda s, x: (calls.append(s.id), _record(s))[1], lambda s: None, resolve,
                     lambda m: None)
    assert calls == ["02_b"] # finished steps skipped, the failed one retried
    assert st["01_a"][0] == "done" and st["02_b"][0] == "ok"
    assert (tmp_path / "02_b.json").exists() and not (tmp_path / "02_b.failed.json").exists()

def test_best_settings(tmp_path):
    steps = {"02_compile_S_default": (kr.Step("02_compile_S_default", "time", "S"), 100.0),
             "02_compile_S_reduce-overhead": (kr.Step("02_compile_S_reduce-overhead", "time", "S", mode="reduce-overhead"), 90.0),
             "03_fp8_S_rowwise_default": (kr.Step("03_fp8_S_rowwise_default", "time", "S", flags="all,fp8", fp8_recipe="rowwise"), 95.0)}
    for k, (s, m) in steps.items():
        kr.write_json(str(tmp_path / f"{k}.json"), _record(s, m))
    res = kr.load_results(str(tmp_path))
    assert kr.best_mode(res, "S")[:2] == ("reduce-overhead", False)
    assert kr.best_overall(res, "S")["fp8_recipe"] == "" # FP8 (95 ms) loses to reduce-overhead (90 ms)
    s = kr.resolve(kr.Step("05_x", "time", "S", mode="@best", fp8_recipe="@best"), res)
    assert s["mode"] == "reduce-overhead" and s["flags"] == "all"
    s = kr.resolve(kr.Step("03_fp8_S_tensorwise_bestmode", "time", "S", flags="all,fp8", fp8_recipe="tensorwise",
                           mode="@best"), res)
    assert s["mode"] == "reduce-overhead" and s["fp8_recipe"] == "tensorwise"

def test_rerank_groups():
    kernels = [("ampere_bf16_s16816gemm_bf16_256x128", 50.0), ("triton_poi_fused_gelu_view_24", 10.0),
               ("triton_red_fused_mul_sum_view_3", 6.0), ("triton_per_fused_add_native_layer_norm_backward_11", 3.0),
               ("triton_per_fused__to_copy_cos_index_mul_native_layer_norm_neg_sin_13", 2.0),
               ("void pytorch_flash::flash_fwd_kernel<...>", 20.0), ("triton_poi_fused_clone_transpose_18", 9.0)]
    shares = kr.group_shares(kernels)
    assert abs(sum(shares.values()) - 100) < 1e-9
    promoted = [p["group"] for p in kr.rerank_from_shares(shares)]
    assert promoted == ["gelu", "transpose_copy", "column_reductions"] # > 5%, largest first; LN (3%) and RoPE stay
    assert kr.kernel_group("triton_red_fused_gelu_gelu_backward_sum_view_9") == "gelu"

def test_write_json_atomic(tmp_path):
    p = str(tmp_path / "x.json")
    kr.write_json(p, {"a": 1})
    assert json.load(open(p)) == {"a": 1} and not os.path.exists(p + ".tmp")

def test_preflight_kernel_family_cudnn_not_flash():
    """ bench/preflight.py: cuDNN SDPA kernel names contain "flash"; they used to be labelled flash """
    import preflight
    cudnn = ["cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_1_128x64x64_4x1x1_cga1x1x1_kernel0_0"]
    assert preflight.kernel_family(cudnn) == "cudnn"
    assert preflight.kernel_family(["void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64>>"]) == "flash"
    assert preflight.kernel_family(["fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention)"]) == "mem_efficient"

def test_cache_families_and_sdpa_isolation(tmp_path):
    s = lambda mode, cd=False: {"mode": mode, "coord_descent": cd}
    # CUDA graphs on/off share; coordinate descent and max-autotune don't mix with the default family
    assert kr.cache_family(s("default")) == kr.cache_family(s("reduce-overhead")) == "default"
    assert kr.cache_family(s("max-autotune")) == kr.cache_family(s("max-autotune-no-cudagraphs")) == "max-autotune"
    assert kr.cache_family(s("max-autotune", True)) == "max-autotune+cd" != kr.cache_family(s("max-autotune"))
    env = kr.cache_env(str(tmp_path), s("max-autotune"), "default")
    assert env == {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor_cache" / "max-autotune")}
    env = kr.cache_env(str(tmp_path), s("max-autotune"), "cudnn") # same autotune results, own compiled graphs
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("max-autotune")
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "0" and env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "0"

def test_kern_steps_pass_flags():
    plan = {s.id: s for s in kr.build_plan(["S"], dry_run=True)}
    assert plan["05_patch_linear_S"].kern == "patch_linear"
    assert plan["05_attn_fused_flex_S"].kern == "attn_fused=flex"
    hw, args = {"capability": [8, 6], "sms": 28}, SimpleNamespace(dry_run=True, data_root=None)
    fields = {"kern_patch_linear", "kern_attn_fused"}
    assert kr.unmet(plan["05_attn_fused_gqa_S"], hw, args, fields) is None # runs in a dry run once config.py has it
    runner = kr.Runner(SimpleNamespace(batch_s=32, batch_b=16, global_batch=None, warmup=3, steps=5, profile_steps=3,
                                       data_root=None, num_workers=4, dry_run=True), hw, "python", "/tmp/x", print)
    assert runner.batch_accum("S") == (32, 2) # dry run: optimizer steps happen within the few steps
    runner.args.dry_run = False
    assert runner.batch_accum("S") == (32, 128) # full run: config's global batch 4096
    st = plan["05_attn_fused_flex_S"]
    cmd = runner.command(st, kr.resolve(st, {}), "/tmp/x/raw")
    assert cmd[cmd.index("--kern") + 1] == "attn_fused=flex"

def test_skip_patterns():
    hw = {"capability": [8, 9], "sms": 142}
    args = SimpleNamespace(dry_run=False, data_root=None, skip=[r"max-autotune\+cd"])
    assert kr.unmet(kr.Step("02_compile_S_max-autotune+cd", "time", "S"), hw, args, set()).startswith("--skip")
    assert kr.unmet(kr.Step("02_compile_S_max-autotune", "time", "S"), hw, args, set()) is None

def test_vanilla_steps_first():
    plan = kr.build_plan(["S", "B"], dry_run=True)
    assert [s.id for s in plan[:3]] == ["00_vanilla_S", "00_vanilla_B", "00_preflight"]
    v = plan[0]
    assert v.kind == "time" and v.flags == "none" and v.mode == "default" and not v.requires # runs in a dry run too
    hw, args = {"capability": [8, 6], "sms": 28}, SimpleNamespace(dry_run=True, data_root=None, skip=[r"vanilla"])
    assert kr.unmet(v, hw, args, set()).startswith("--skip") # honors --skip like every step
    runner = kr.Runner(SimpleNamespace(batch_s=32, batch_b=16, global_batch=None, warmup=3, steps=5, profile_steps=3,
                                       data_root=None, num_workers=4, dry_run=True), hw, "python", "/tmp/x", print)
    cmd = runner.command(v, kr.resolve(v, {}), "/tmp/x/raw")
    assert cmd[cmd.index("--flags") + 1] == "none" and cmd[cmd.index("--mode") + 1] == "default"
    assert "--kern" not in cmd and "--fp8-recipe" not in cmd

def _rec(flags, ms_, loss, status="ok"):
    return {"status": status, "settings": {"flags": flags, "mode": "default"},
            "result": {"ms_per_step_median": ms_, "loss_last": loss}}

def test_loss_and_speed_vs_vanilla():
    step = kr.Step("05_x_S", "time", "S")
    res = {"00_vanilla_S": _rec("none", 120.0, 1.0)}
    loss, rel, speed, flag, why = kr.vs_vanilla(step, _rec("all", 100.0, 1.005), res)
    assert loss == 1.005 and abs(rel - 0.005) < 1e-12 and abs(speed - 1.2) < 1e-12 and not flag and why == ""
    assert kr.vs_vanilla(step, _rec("all", 100.0, 1.02), res)[3] # 2% > 1% for non-FP8
    assert not kr.vs_vanilla(step, _rec("all,fp8", 100.0, 1.02), res)[3] # within FP8's 5%
    assert kr.vs_vanilla(step, _rec("all,fp8", 100.0, 1.06), res)[3]
    # vanilla missing or failed: n/a, no crash
    for r in ({}, {"00_vanilla_S": _rec("none", 120.0, 1.0, status="failed")}):
        assert kr.vs_vanilla(step, _rec("all", 100.0, 1.0), r) == (1.0, None, None, False, "")
    prof = kr.Step("04_profile_S_best", "profile", "S") # profile steps have no loss / step time to compare
    assert kr.vs_vanilla(prof, {"status": "ok", "result": {}}, res) == (None, None, None, False, "")

def test_summary_markdown_columns_without_vanilla():
    plan = [kr.Step("00_vanilla_S", "time", "S", flags="none"), kr.Step("02_compile_S_default", "time", "S")]
    rows = kr.summary_rows(plan, {"02_compile_S_default": _rec("all", 100.0, 1.0)}, {"00_vanilla_S": ("failed", "x")})
    md = kr.summary_markdown(rows, {"name": "GPU", "capability": [8, 9], "sms": 142},
                             SimpleNamespace(run_id="t", gpu="0", dry_run=True, models=["S"], warmup=1, steps=1), {}, 1.0)
    assert "| speedup vs vanilla | loss | loss vs vanilla |" in md and "n/a" in md and "Loss flags: none." in md

def test_synthetic_inputs_same_with_and_without_uint8():
    """ bench/harness.py: the vanilla step (no perf_uint8_images) and the flagged steps get the same images, so their
    losses compare; LitREPA.on_after_batch_transfer normalizes the uint8 ones """
    import torch
    import harness
    from repa import LitREPA
    from test_perf_equivalence import make_config
    ref, new = make_config("tiny"), make_config("tiny", "uint8_images")
    a, b = harness._SyntheticBatches(ref, 1).batch, harness._SyntheticBatches(new, 1).batch
    assert b["img"].dtype == torch.uint8 and torch.equal(a["position_ids"], b["position_ids"])
    out = LitREPA(new).on_after_batch_transfer({"img": b["img"].clone()}, 0)["img"]
    assert torch.equal(out, a["img"])

def test_real_data_loss_not_compared_with_vanilla():
    """ vanilla trains on synthetic images: a real-data row's loss differs for that reason alone (smoke run: -1.19%) """
    res = {"00_vanilla_S": _rec("none", 120.0, 1.0102)}
    real = kr.Step("06_data_S_real", "time", "S", data="real")
    loss, rel, speed, flag, why = kr.vs_vanilla(real, _rec("all", 100.0, 0.9981), res)
    assert loss == 0.9981 and rel is None and not flag and why == "real data" and speed is not None
    rows = kr.summary_rows([real], {"06_data_S_real": {**_rec("all", 100.0, 0.9981), "kind": "time"}}, {})
    md = kr.summary_markdown(rows, {"name": "GPU", "capability": [8, 9], "sms": 142},
                             SimpleNamespace(run_id="t", gpu="0", dry_run=False, models=["S"], warmup=1, steps=1), {}, 1.0)
    assert "n/a (real data)" in md and "Loss flags: none." in md

def test_non_finite_loss_fails_the_step():
    step = kr.Step("02_compile_S_default", "time", "S")
    assert kr.classify(step, {"mode": "default"}, {"loss_all_finite": True}) == ("ok", None)
    status, err = kr.classify(step, {"mode": "default"}, {"loss_all_finite": False})
    assert status == "failed" and err == "non-finite loss"
    fp8 = kr.Step("03_fp8_S_tensorwise_bestmode", "time", "S")
    status, err = kr.classify(fp8, {"mode": "max-autotune", "fp8_recipe": "tensorwise"}, {"loss_all_finite": False})
    assert status == "failed" and "known issue: FP8 tensorwise under max-autotune" in err
    prof = kr.Step("04_profile_S_best", "profile", "S") # profiles record no loss
    assert kr.classify(prof, {"mode": "default"}, {}) == ("ok", None)
    assert kr.classify(step, {"mode": "default"}, None)[0] == "failed"

def test_resume_reruns_ok_step_with_non_finite_loss(tmp_path):
    """ results written before the rule: status ok but a NaN loss (smoke run, 03_fp8_S_tensorwise_bestmode). Resume
    must redo it, not skip it, and it must never count as the best mode """
    step = kr.Step("03_fp8_S_tensorwise_bestmode", "time", "S", flags="all,fp8", fp8_recipe="tensorwise")
    nan_rec = {**_record(step, 90.0), "kind": "time", "result": {"ms_per_step_median": 90.0, "loss_all_finite": False}}
    kr.write_json(str(tmp_path / f"{step.id}.json"), nan_rec)
    assert not kr.finite_ok(nan_rec)
    ran = []
    resolve = lambda st, results: {"mode": "max-autotune", "coord_descent": False, "fp8_recipe": "tensorwise", "flags": "all,fp8"}
    states = kr.run_plan([step], str(tmp_path), lambda st, x: (ran.append(st.id), _record(st))[1], lambda st: None,
                         resolve, lambda m: None)
    assert ran == [step.id] and states[step.id][0] == "ok"
    fast_nan = {**nan_rec, "step": "02_compile_S_max-autotune", "settings": {"mode": "max-autotune", "coord_descent": False}}
    ok = {**_rec("all", 100.0, 1.0), "kind": "time", "settings": {"mode": "default", "coord_descent": False}}
    assert kr.best_mode({"02_compile_S_max-autotune": fast_nan, "02_compile_S_default": ok}, "S")[0] == "default"

def test_attention_steps_run_without_cuda_graphs():
    """ SDPA cudnn / efficient and attn_fused=flex failed CUDA graph capture under max-autotune on the RTX 6000 Ada:
    they run the no-CUDA-graph twin of the best mode and compare speed with the matching 02 step """
    plan = {s.id: s for s in kr.build_plan(["S"], dry_run=False)}
    assert all(plan[i].no_cudagraphs for i in ("05_sdpa_S_cudnn", "05_sdpa_S_efficient", "05_attn_fused_flex_S"))
    assert not any(plan[i].no_cudagraphs for i in ("05_patch_linear_S", "05_attn_fused_gqa_S", "06_data_S_synthetic"))
    results = {"02_compile_S_max-autotune": {**_rec("all", 199.2, 1.0), "kind": "time",
                                             "settings": {"mode": "max-autotune", "coord_descent": False}},
               "02_compile_S_max-autotune-no-cudagraphs": {**_rec("all", 202.5, 1.0), "kind": "time",
                                                           "settings": {"mode": "max-autotune-no-cudagraphs",
                                                                        "coord_descent": False}}}
    s = kr.resolve(plan["05_attn_fused_flex_S"], results)
    assert s["mode"] == "max-autotune-no-cudagraphs" and s["cudagraphs_off_from"] == "max-autotune"
    assert kr.resolve(plan["05_patch_linear_S"], results)["mode"] == "max-autotune" # unchanged
    rec = {"status": "ok", "kind": "time", "settings": s, "result": {"ms_per_step_median": 200.0, "loss_last": 1.0}}
    row = kr.summary_rows([plan["05_attn_fused_flex_S"]], {**results, "05_attn_fused_flex_S": rec}, {})[0]
    assert abs(row[7] - 202.5 / 200.0) < 1e-12 # vs 02_compile_S_max-autotune-no-cudagraphs, not 02 default
    assert "CUDA graphs off" in row[9] and "02_compile_S_max-autotune-no-cudagraphs" in row[9]
    s = kr.resolve(plan["05_sdpa_S_cudnn"], {"02_compile_S_reduce-overhead": {**results["02_compile_S_max-autotune"],
                                              "settings": {"mode": "reduce-overhead", "coord_descent": False}}})
    assert s["mode"] == "default" and s["cudagraphs_off_from"] == "reduce-overhead"
