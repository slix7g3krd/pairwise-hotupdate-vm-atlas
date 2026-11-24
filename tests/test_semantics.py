"""语义测试：逐条覆盖题目要求的六项不变量。"""

import gameplay.hotupdate_vm_atlas as mod
from gameplay.hotupdate_vm_atlas import (
    MISSING,
    AtlasRef,
    HotUpdateRuntime,
)


def _runtime(gen=1, atlases=("hero", "ui")):
    runtime = HotUpdateRuntime()
    runtime.register(gen, atlases=atlases)
    runtime.install(gen, verified=True)
    runtime.activate(gen)
    return runtime


# 1. 闭包与资源代际绑定 -------------------------------------------------------


def test_closure_stays_bound_to_its_generation_after_upgrade():
    runtime = _runtime(1)
    old = runtime.make_closure(1, "hero")
    before = old()

    runtime.register(2, atlases=("hero", "ui", "fx"))
    runtime.install(2, verified=True)
    runtime.activate(2)

    after = old()
    assert isinstance(before, AtlasRef) and before.generation == 1
    assert isinstance(after, AtlasRef) and after.generation == 1
    assert after.cache_key() == "g1/hero"
    # 旧代际仍被闭包租约钉住，没有指向已被卸载/替换的图集。
    assert runtime.is_loaded(1)


def test_closure_never_resolves_into_a_newer_generation():
    runtime = _runtime(1)
    closure = runtime.make_closure(1, "hero")
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)

    ref = closure()
    assert isinstance(ref, AtlasRef)
    assert ref.generation != 2


def test_released_generation_unloads_and_old_closure_returns_missing():
    runtime = _runtime(1)
    closure = runtime.make_closure(1, "hero")
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)

    closure.dispose()
    # 没有租约、也不是激活代际 → 立刻回收。
    assert not runtime.is_loaded(1)
    # 已释放的闭包不返回悬空引用，而是安全缺失。
    assert closure() is MISSING


# 2. 校验完成门控 -------------------------------------------------------------


def test_download_completion_is_not_activation_permission():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.begin_update(2)
    assert runtime.finish_download(2) == "verifying"

    result = runtime.activate(2)
    assert result["activated"] is False
    assert result["reason"] == "verification_pending"
    assert runtime.active == 1


def test_activation_allowed_only_after_successful_verification():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=True)

    result = runtime.activate(2)
    assert result["activated"] is True
    assert runtime.active == 2


def test_failed_verification_keeps_gate_shut():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=False)

    assert runtime.activate(2)["activated"] is False
    assert runtime.active == 1
    assert not runtime.is_loaded(2)


# 3. 更新期场景隔离 -----------------------------------------------------------


def test_activation_is_deferred_while_a_scene_is_alive():
    runtime = _runtime(1)
    runtime.enter_scene("battle", battle=True)
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)

    result = runtime.activate(2)
    assert result["activated"] is False
    assert result["deferred"] is True
    assert runtime.active == 1
    assert runtime.scene_generation("battle") == 1


def test_scene_keeps_one_generation_for_its_whole_life():
    runtime = _runtime(1)
    runtime.enter_scene("battle", battle=True)
    closure = runtime.make_closure(runtime.scene_generation("battle"), "hero")
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)

    assert closure().generation == 1
    assert runtime.metrics_snapshot()["mixed_version_events"] == 0


def test_pending_activation_flushes_after_last_scene_exits():
    runtime = _runtime(1)
    runtime.enter_scene("battle", battle=True)
    runtime.enter_scene("lobby")
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)

    first = runtime.exit_scene("battle")
    assert first["flushed"] is None
    assert runtime.active == 1

    second = runtime.exit_scene("lobby")
    assert second["flushed"]["activated"] is True
    assert runtime.active == 2


# 4. 回滚清缓存 ---------------------------------------------------------------


def test_rollback_clears_closure_registry_and_generation_cache():
    runtime = _runtime(1)
    runtime.make_closure(1, "hero")
    runtime.cache_put(1, "hero", b"gen1")

    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)
    runtime.make_closure(2, "hero")
    runtime.cache_put(2, "hero", b"gen2")

    result = runtime.rollback("runtime_error")

    assert result["rolled_back"] is True
    assert runtime.active == 1
    assert runtime.closure_count() == 0
    assert runtime.cache_keys() == []
    assert not runtime.is_loaded(2)

    # 回滚后新建的闭包只解析到旧版本。
    fresh = runtime.make_closure(runtime.active, "hero")()
    assert isinstance(fresh, AtlasRef) and fresh.generation == 1


def test_rollback_deferred_while_battle_is_active():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)
    runtime.enter_scene("battle", battle=True)

    result = runtime.rollback("crash")
    assert result["rolled_back"] is False
    assert result["deferred"] is True
    assert runtime.active == 2
    # 场景内代际不动，避免战场用着失败版本的闭包。
    assert runtime.scene_generation("battle") == 2


def test_deferred_rollback_applies_after_scene_exits():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)
    runtime.enter_scene("battle", battle=True)
    runtime.rollback("crash")

    exit_result = runtime.exit_scene("battle")
    assert exit_result["flushed_rollback"]["rolled_back"] is True
    assert runtime.active == 1
    assert runtime.closure_count() == 0
    assert runtime.cache_keys() == []


# 5. AB 包版本一致 ------------------------------------------------------------


def test_ab_version_mismatch_blocks_activation():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",), code_version="2.0.0", resource_version="1.9.0")
    runtime.install(2, verified=True)

    result = runtime.activate(2)
    assert result["activated"] is False
    assert result["reason"] == "ab_version_mismatch"
    assert runtime.active == 1


def test_ab_version_match_allows_activation():
    runtime = _runtime(1)
    runtime.register(3, atlases=("hero",), code_version="3.0.0", resource_version="3.0.0")
    runtime.install(3, verified=True)

    assert runtime.activate(3)["activated"] is True


# 6. 失败回退旧版本 -----------------------------------------------------------


def test_failed_update_falls_back_and_old_closure_still_works():
    runtime = _runtime(1)
    old = runtime.make_closure(1, "hero")
    runtime.register(2, atlases=("hero",))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=False)
    runtime.activate(2)

    result = runtime.fallback_to_previous(2, reason="verification_failed")

    assert result["fallback"] is True
    assert runtime.active == 1
    assert not runtime.is_loaded(2)
    ref = old()
    assert isinstance(ref, AtlasRef) and ref.generation == 1


def test_activation_failure_rolls_back_to_previous_good():
    runtime = _runtime(1)
    runtime.register(2, atlases=("hero",))
    runtime.install(2, verified=True)
    runtime.activate(2)

    result = runtime.rollback("activation_error")
    assert result["rolled_back"] is True
    assert runtime.active == 1
    assert runtime.is_loaded(1)


# 交叉不变量 ------------------------------------------------------------------


def test_audit_has_no_violations():
    report = mod.audit(1)
    assert report["ok"] is True
    assert report["violations"] == []


def test_metrics_report_no_mixed_version_events():
    report = mod.metrics(1)
    assert report["ok"] is True
    assert report["mixed_version_events"] == 0


def test_all_reports_are_internally_consistent():
    for name in (
        "closure_generation",
        "verify_gate",
        "scene_isolate",
        "rollback_clear",
        "ab_version",
        "old_fallback",
        "atlas_ref",
        "validation",
        "audit",
        "fallback",
        "download_state",
        "metrics",
        "mid_battle",
        "cache_key",
        "determinism",
    ):
        report = getattr(mod, name)(1)
        assert report["ok"] is True, name
        assert not report.get("problems", []), name


def test_determinism_is_repeatable():
    assert mod.determinism(7) == mod.determinism(7)
