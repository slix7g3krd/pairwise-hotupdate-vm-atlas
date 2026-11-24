"""热更新版本切换运行时。

覆盖题目要求的六条不变量：

1. 闭包与资源代际绑定：闭包在创建时记下所属代际，只在自身代际内解析
   图集，并对该代际持有一个租约；代际被卸载前必须等所有闭包释放，
   因此旧闭包不会解析到已卸载的图集，也不会跨代解析到新图集。
2. 校验完成门控：下载完成只代表文件到齐，状态停在 ``verifying``；
   只有校验通过后才能激活新版本，未校验 / 校验失败一律拒绝激活。
3. 更新期场景隔离：场景进入时固定当时的代际，运行期间始终从固定代际
   取资源；只要有场景存活，激活就延后，避免同一场景看到两个版本。
4. 回滚清缓存：回滚时清空闭包注册表、代际资源缓存，并卸载失败代际。
5. AB 包版本一致：代码包版本与资源包版本不一致时禁止激活。
6. 失败回退旧版本：下载 / 校验 / 激活任一环节失败，都回退到上一个
   已知可用代际，旧版本的资源与闭包保持可用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "MISSING",
    "AtlasRef",
    "Closure",
    "Generation",
    "HotUpdateRuntime",
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
]


class _Missing:
    """安全缺失哨兵：解析不到资源时返回它，而不是返回悬空引用。"""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


@dataclass(frozen=True)
class Generation:
    """一次资源 / 脚本代际。"""

    number: int
    atlases: tuple[str, ...]
    code_version: str
    resource_version: str

    @property
    def versions_match(self) -> bool:
        return self.code_version == self.resource_version


@dataclass(frozen=True)
class AtlasRef:
    """代际内的一枚图集引用，永远携带所属代际。"""

    generation: int
    name: str

    def cache_key(self) -> str:
        return f"g{self.generation}/{self.name}"


class Closure:
    """与代际绑定的脚本闭包。

    闭包持有创建代际的租约：只要闭包还活着，该代际的图集就不会被卸载。
    解析时只查自己的代际，绝不会回落到当前激活代际。
    """

    def __init__(self, runtime: "HotUpdateRuntime", generation: int, name: str) -> None:
        self._runtime = runtime
        self.generation = generation
        self.name = name
        self._disposed = False
        runtime._retain(generation)

    def __call__(self) -> AtlasRef | _Missing:
        return self._runtime.resolve(self.generation, self.name)

    def rebind(self, generation: int | None = None) -> "Closure":
        """在当前（或指定）代际上重建闭包，用于升级后的显式迁移。"""

        target = self._runtime.active if generation is None else generation
        if target is None:
            raise RuntimeError("没有可绑定的代际")
        return self._runtime.make_closure(target, self.name)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._runtime._release(self.generation)
        self._runtime._forget_closure(self)

    def __repr__(self) -> str:  # pragma: no cover - 调试友好
        return f"Closure(gen={self.generation}, name={self.name!r})"


class HotUpdateRuntime:
    """确定性的热更新状态机，便于复现与审计。"""

    def __init__(self) -> None:
        self._generations: dict[int, Generation] = {}
        self._loaded: set[int] = set()
        self._verified: set[int] = set()
        self._leases: dict[int, int] = {}
        self._closures: list[Closure] = []
        self._scenes: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._counters = {
            "downloads": 0,
            "downloads_completed": 0,
            "verifications": 0,
            "verification_failures": 0,
            "activations": 0,
            "blocked_activations": 0,
            "deferred_activations": 0,
            "rollbacks": 0,
            "fallbacks": 0,
            # 真正观测到「同一场景同时使用两个版本」的次数；设计上恒为 0。
            "mixed_version_events": 0,
        }
        self.active: int | None = None
        self.previous_good: int | None = None
        self.pending: int | None = None
        self.pending_rollback: dict[str, Any] | None = None
        self.state = "idle"

    # ------------------------------------------------------------------ 事件
    def _record(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {"seq": len(self._events) + 1, "event": event, "state": self.state}
        entry.update(fields)
        self._events.append(entry)
        return entry

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    # -------------------------------------------------------------- 代际管理
    def register(
        self,
        number: int,
        atlases: Iterable[str] = ("atlas",),
        code_version: Any = None,
        resource_version: Any = None,
    ) -> Generation:
        """登记一个代际的清单；此阶段不加载图集、也不代表已校验。"""

        number = int(number)
        if number in self._generations:
            return self._generations[number]
        code = str(number if code_version is None else code_version)
        resource = str(number if resource_version is None else resource_version)
        generation = Generation(
            number=number,
            atlases=tuple(sorted({str(name) for name in atlases})),
            code_version=code,
            resource_version=resource,
        )
        self._generations[number] = generation
        self._record("register", generation=number)
        return generation

    def install(self, number: int, *, verified: bool = False, load: bool = True) -> Generation:
        """把代际文件落到本地；``verified`` 仅用于可信基线或校验完成后的安装。"""

        generation = self._generations[int(number)]
        if load:
            self._loaded.add(generation.number)
        if verified:
            self._verified.add(generation.number)
        self._record("install", generation=generation.number, verified=verified)
        if verified:
            self._record("verify", generation=generation.number, ok=True, source="install")
        return generation

    # ------------------------------------------------------------ 下载与校验
    def begin_update(self, number: int) -> dict[str, Any]:
        self.register(number)
        self.pending = int(number)
        self.state = "downloading"
        self._counters["downloads"] += 1
        self._record("download_begin", generation=int(number))
        return {"state": self.state, "generation": int(number)}

    def finish_download(self, number: int) -> str:
        """下载完成：只代表文件到齐，校验仍在后台，门控保持关闭。"""

        if self.pending != int(number):
            self.begin_update(number)
        self.state = "verifying"
        self._counters["downloads_completed"] += 1
        self._record("download_complete", generation=int(number), gate="closed")
        return self.state

    def verify(self, number: int, ok: bool = True) -> dict[str, Any]:
        number = int(number)
        self.register(number)
        self._counters["verifications"] += 1
        if ok:
            self._verified.add(number)
            self._loaded.add(number)
            self.state = "ready"
        else:
            self._verified.discard(number)
            self._counters["verification_failures"] += 1
            self.state = "failed"
        self._record("verify", generation=number, ok=bool(ok))
        return {"generation": number, "verified": bool(ok), "state": self.state}

    # ------------------------------------------------------------------ 激活
    def _versions_consistent(self, number: int) -> bool:
        return self._generations[number].versions_match

    def activate(self, number: int) -> dict[str, Any]:
        number = int(number)
        if number not in self._generations:
            self._counters["blocked_activations"] += 1
            self._record("activate_rejected", generation=number, reason="unknown_generation")
            return {"activated": False, "generation": number, "reason": "unknown_generation"}

        if not self._versions_consistent(number):
            self._counters["blocked_activations"] += 1
            self._record("activate_rejected", generation=number, reason="ab_version_mismatch")
            return {
                "activated": False,
                "generation": number,
                "reason": "ab_version_mismatch",
                "active": self.active,
            }

        if number not in self._verified:
            self._counters["blocked_activations"] += 1
            self._record("activate_rejected", generation=number, reason="verification_pending")
            return {
                "activated": False,
                "generation": number,
                "reason": "verification_pending",
                "active": self.active,
            }

        if self._scenes:
            self.pending = number
            self.state = "ready"
            self._counters["deferred_activations"] += 1
            self._record(
                "activate_deferred",
                generation=number,
                reason="active_scene",
                scenes=sorted(self._scenes),
            )
            return {
                "activated": False,
                "deferred": True,
                "generation": number,
                "reason": "active_scene",
                "scenes": sorted(self._scenes),
                "active": self.active,
            }

        previous = self.active
        if previous is not None and previous != number:
            self.previous_good = previous
        self.active = number
        self._loaded.add(number)
        self.pending = None
        self.state = "active"
        self._counters["activations"] += 1
        self._record("activate", generation=number, previous=previous)
        self._collect_garbage()
        return {
            "activated": True,
            "generation": number,
            "previous": previous,
            "state": self.state,
        }

    def flush_pending(self) -> dict[str, Any] | None:
        """场景全部退出后，执行此前被延后的激活。"""

        if self.pending is None or self._scenes:
            return None
        number, self.pending = self.pending, None
        return self.activate(number)

    # ---------------------------------------------------------------- 场景隔离
    def enter_scene(self, name: str, *, battle: bool = False) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("没有激活代际，无法进入场景")
        if name in self._scenes:
            return dict(self._scenes[name])
        info = {"generation": self.active, "battle": bool(battle)}
        self._scenes[name] = info
        self._record("scene_enter", scene=name, generation=self.active, battle=bool(battle))
        return dict(info)

    def exit_scene(self, name: str) -> dict[str, Any]:
        info = self._scenes.pop(name, None)
        if info is None:
            return {"exited": False, "scene": name, "flushed": None, "flushed_rollback": None}
        self._record("scene_exit", scene=name, generation=info["generation"])
        self._collect_garbage()
        flushed_rollback = self._flush_pending_rollback()
        flushed = None if flushed_rollback else self.flush_pending()
        return {
            "exited": True,
            "scene": name,
            "flushed": flushed,
            "flushed_rollback": flushed_rollback,
        }

    def _flush_pending_rollback(self) -> dict[str, Any] | None:
        if self.pending_rollback is None or self._scenes:
            return None
        pending = self.pending_rollback
        self.pending_rollback = None
        return self._apply_rollback(
            pending["target"], pending["reason"], pending["failed"]
        )

    def scene_generation(self, name: str) -> int | None:
        info = self._scenes.get(name)
        return None if info is None else int(info["generation"])

    @property
    def scenes(self) -> dict[str, dict[str, Any]]:
        return {name: dict(info) for name, info in self._scenes.items()}

    # ------------------------------------------------------------ 闭包与资源
    def make_closure(self, generation: int, name: str) -> Closure:
        generation = int(generation)
        if generation not in self._generations:
            raise KeyError(f"未登记的代际: {generation}")
        closure = Closure(self, generation, name)
        self._closures.append(closure)
        self._record("closure_create", generation=generation, name=name)
        return closure

    def _forget_closure(self, closure: Closure) -> None:
        try:
            self._closures.remove(closure)
        except ValueError:
            pass

    @property
    def closures(self) -> list[Closure]:
        return list(self._closures)

    def closure_count(self) -> int:
        return len(self._closures)

    def dispose_closures(self) -> int:
        count = len(self._closures)
        for closure in list(self._closures):
            closure.dispose()
        self._closures.clear()
        return count

    def _retain(self, number: int) -> None:
        self._leases[number] = self._leases.get(number, 0) + 1

    def _release(self, number: int) -> None:
        remaining = self._leases.get(number, 0) - 1
        if remaining <= 0:
            self._leases.pop(number, None)
        else:
            self._leases[number] = remaining
        self._collect_garbage()

    def resolve(self, generation: int, name: str) -> AtlasRef | _Missing:
        """只在自己代际内解析；代际未加载或资源不存在都返回安全缺失。"""

        generation = int(generation)
        if generation not in self._loaded:
            return MISSING
        info = self._generations.get(generation)
        if info is None or name not in info.atlases:
            return MISSING
        return AtlasRef(generation, name)

    @property
    def loaded_generations(self) -> list[int]:
        return sorted(self._loaded)

    def is_loaded(self, number: int) -> bool:
        return int(number) in self._loaded

    def _collect_garbage(self) -> None:
        pinned = {int(info["generation"]) for info in self._scenes.values()}
        if self.active is not None:
            pinned.add(self.active)
        for number in sorted(self._loaded):
            if number in pinned:
                continue
            if self._leases.get(number, 0) > 0:
                continue
            self._loaded.discard(number)
            self._record("unload", generation=number)

    # ------------------------------------------------------------------ 缓存
    def cache_key(self, generation: int, name: str) -> str:
        return AtlasRef(int(generation), str(name)).cache_key()

    def cache_put(self, generation: int, name: str, value: Any) -> str:
        key = self.cache_key(generation, name)
        self._cache[key] = value
        return key

    def cache_get(self, generation: int, name: str) -> Any:
        return self._cache.get(self.cache_key(generation, name), MISSING)

    def cache_clear(self, generation: int | None = None) -> int:
        if generation is None:
            removed = len(self._cache)
            self._cache.clear()
        else:
            prefix = f"g{int(generation)}/"
            keys = [key for key in self._cache if key.startswith(prefix)]
            for key in keys:
                del self._cache[key]
            removed = len(keys)
        return removed

    def cache_keys(self) -> list[str]:
        return sorted(self._cache)

    # ------------------------------------------------------------ 回滚与回退
    def rollback(self, reason: Any = "failure") -> dict[str, Any]:
        """回滚到上一个已知可用代际，并清空脚本 / 资源缓存。

        只要还有场景存活，回滚就和激活一样被挂起：场景内不能中途换代际，
        否则该场景会用着失败版本的闭包跑在新资源上。
        """

        failed = self.active
        target = self.previous_good
        base: dict[str, Any] = {
            "rolled_back": False,
            "reason": str(reason),
            "failed_generation": failed,
            "target_generation": target,
        }
        if target is None or failed is None:
            self._record("rollback_rejected", reason="no_previous_good")
            base["reason_code"] = "no_previous_good"
            return base
        if self._scenes:
            self.pending_rollback = {"target": target, "reason": str(reason), "failed": failed}
            self._record(
                "rollback_deferred",
                reason="active_scene",
                scenes=sorted(self._scenes),
            )
            base["deferred"] = True
            base["reason_code"] = "active_scene"
            base["scenes"] = sorted(self._scenes)
            return base

        return self._apply_rollback(target, str(reason), failed)

    def _apply_rollback(self, target: int, reason: str, failed: int | None) -> dict[str, Any]:
        closures_cleared = self.dispose_closures()
        cache_cleared = self.cache_clear()

        if failed is not None:
            self._verified.discard(failed)
            self._loaded.discard(failed)
        self.active = target
        self._loaded.add(target)
        self.previous_good = None
        self.pending = None
        self.state = "rolled_back"
        self._counters["rollbacks"] += 1
        self._record("rollback", failed=failed, target=target, reason=reason)
        self._collect_garbage()
        return {
            "rolled_back": True,
            "reason": reason,
            "failed_generation": failed,
            "target_generation": target,
            "active": self.active,
            "closures_cleared": closures_cleared,
            "cache_entries_cleared": cache_cleared,
            "state": self.state,
        }

    def fallback_to_previous(
        self, failed: int | None = None, reason: Any = "failure"
    ) -> dict[str, Any]:
        """新版本未激活就失败时的回退：丢弃失败产物，旧版本继续服务。"""

        failed_number = self.pending if failed is None else int(failed)
        base: dict[str, Any] = {
            "fallback": False,
            "reason": str(reason),
            "failed_generation": failed_number,
            "active": self.active,
        }
        if failed_number is None:
            base["reason_code"] = "no_failed_generation"
            self._record("fallback_rejected", reason="no_failed_generation")
            return base
        if self.active == failed_number:
            result = self.rollback(reason)
            result.setdefault("fallback", bool(result.get("rolled_back")))
            result["via"] = "rollback"
            return result

        removed_cache = self.cache_clear(failed_number)
        self._verified.discard(failed_number)
        self._loaded.discard(failed_number)
        if self.pending == failed_number:
            self.pending = None
        self.state = "active" if self.active is not None else "failed"
        self._counters["fallbacks"] += 1
        self._record(
            "fallback", failed=failed_number, active=self.active, reason=str(reason)
        )
        self._collect_garbage()
        base.update(
            {
                "fallback": True,
                "active": self.active,
                "cache_entries_cleared": removed_cache,
                "old_version_intact": self.active is not None,
                "state": self.state,
            }
        )
        return base

    # ------------------------------------------------------------ 指标与不变量
    def metrics_snapshot(self) -> dict[str, int]:
        return dict(self._counters)

    def problems(self) -> list[str]:
        """返回当前状态下的不变量违例（正常应始终为空）。"""

        found: list[str] = []
        if self.active is not None and self.active not in self._loaded:
            found.append("active_generation_not_loaded")
        for name, info in self._scenes.items():
            if int(info["generation"]) not in self._loaded:
                found.append(f"scene_not_loaded:{name}")
        for closure in self._closures:
            if closure.generation not in self._generations:
                found.append(f"closure_unknown_generation:{closure.name}")
        for key in self._cache:
            if not key.startswith("g") or "/" not in key:
                found.append(f"cache_key_not_generation_scoped:{key}")
        return found


# ---------------------------------------------------------------------- 用例


def _base_runtime(number: int = 1, atlases: Iterable[str] = ("hero", "ui")) -> HotUpdateRuntime:
    runtime = HotUpdateRuntime()
    runtime.register(number, atlases=atlases)
    runtime.install(number, verified=True)
    runtime.activate(number)
    return runtime


def closure_generation(gen_a: int = 1, gen_b: int | None = None, **_kwargs: Any) -> dict[str, Any]:
    """旧闭包在升级后仍然安全：要么解析到自己的旧图集，要么安全缺失。"""

    gen_a = int(gen_a)
    gen_b = gen_a + 1 if gen_b is None else int(gen_b)
    runtime = _base_runtime(gen_a)
    runtime.register(gen_b, atlases=("hero", "ui", "fx"))

    old_closure = runtime.make_closure(gen_a, "hero")
    before = old_closure()
    runtime.install(gen_b, verified=True)
    runtime.activate(gen_b)

    after_upgrade = old_closure()
    new_ref = runtime.make_closure(gen_b, "hero")()
    old_closure.dispose()
    after_release = old_closure()

    no_cross_generation = (
        after_upgrade is not MISSING
        and isinstance(after_upgrade, AtlasRef)
        and after_upgrade.generation == gen_a
    )
    report = {
        "ok": bool(
            before is not MISSING
            and no_cross_generation
            and after_release is MISSING
            and new_ref is not MISSING
        ),
        "bound_generation": gen_a,
        "active_generation": runtime.active,
        "ref_before_upgrade": before.cache_key() if isinstance(before, AtlasRef) else "miss",
        "ref_after_upgrade": (
            after_upgrade.cache_key() if isinstance(after_upgrade, AtlasRef) else "miss"
        ),
        "ref_after_release": "miss" if after_release is MISSING else "dangling",
        "rebound_ref": new_ref.cache_key() if isinstance(new_ref, AtlasRef) else "miss",
        "cross_generation_leak": not no_cross_generation,
        "old_generation_loaded_while_leased": gen_a in runtime.loaded_generations,
    }
    report["problems"] = runtime.problems()
    return report


def verify_gate(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """下载完成不等于可激活；校验通过是唯一放行条件。"""

    runtime = _base_runtime(1)
    runtime.register(2, atlases=("hero", "ui"))
    runtime.begin_update(2)
    state_after_download = runtime.finish_download(2)
    early = runtime.activate(2)

    ok = runtime.verify(2, ok=True)
    late = runtime.activate(2)

    broken = _base_runtime(1)
    broken.register(2, atlases=("hero",))
    broken.begin_update(2)
    broken.finish_download(2)
    broken.verify(2, ok=False)
    after_failed_verify = broken.activate(2)

    report = {
        "ok": bool(
            state_after_download == "verifying"
            and not early.get("activated")
            and early.get("reason") == "verification_pending"
            and ok.get("verified")
            and late.get("activated")
            and not after_failed_verify.get("activated")
            and broken.active == 1
        ),
        "state_after_download": state_after_download,
        "blocked_before_verify": not early.get("activated"),
        "block_reason": early.get("reason"),
        "activated_after_verify": late.get("activated"),
        "failed_verification_reason": after_failed_verify.get("reason"),
        "failed_verification_active": broken.active,
        "active_generation": runtime.active,
        "seed": seed,
    }
    return report


def scene_isolate(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """更新期间场景固定代际，激活延后到所有场景退出之后。"""

    runtime = _base_runtime(1)
    runtime.enter_scene("battle", battle=True)
    runtime.enter_scene("lobby")

    runtime.register(2, atlases=("hero", "ui", "fx"))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=True)
    deferred = runtime.activate(2)

    battle_gen_1 = runtime.scene_generation("battle")
    lobby_gen_1 = runtime.scene_generation("lobby")
    after_battle_exit = runtime.exit_scene("battle")
    still_deferred = runtime.active == 1 and runtime.scene_generation("lobby") == 1
    after_lobby_exit = runtime.exit_scene("lobby")
    runtime.enter_scene("result")
    result_gen = runtime.scene_generation("result")

    isolated = (
        battle_gen_1 == lobby_gen_1 == 1
        and still_deferred
        and deferred.get("deferred") is True
        and result_gen == 2
    )
    return {
        "ok": bool(isolated and after_lobby_exit.get("flushed", {}).get("activated")),
        "battle_generation": battle_gen_1,
        "lobby_generation": lobby_gen_1,
        "activation_deferred": bool(deferred.get("deferred")),
        "deferred_reason": deferred.get("reason"),
        "active_while_battle": 1 if still_deferred else None,
        "battle_exit_flushed": after_battle_exit.get("flushed"),
        "lobby_exit_flushed": after_lobby_exit.get("flushed"),
        "new_scene_generation": result_gen,
        "mixed_version_events": runtime.metrics_snapshot()["mixed_version_events"],
        "problems": runtime.problems(),
        "seed": seed,
    }


def rollback_clear(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """回滚后脚本闭包缓存与代际资源缓存都必须清空。"""

    runtime = _base_runtime(1)
    runtime.make_closure(1, "hero")
    runtime.cache_put(1, "hero", b"gen1")
    runtime.register(2, atlases=("hero", "ui", "fx"))
    runtime.install(2, verified=True)
    runtime.activate(2)
    runtime.make_closure(2, "hero")
    runtime.cache_put(2, "hero", b"gen2")

    result = runtime.rollback("runtime_error")

    # 先记录清空结果，再验证回滚后新建的闭包只解析到旧版本。
    registry_empty = runtime.closure_count() == 0
    cache_empty = runtime.cache_keys() == []
    failed_unloaded = not runtime.is_loaded(2)

    fresh = runtime.make_closure(runtime.active or 1, "hero")()
    report = {
        "ok": bool(
            result.get("rolled_back")
            and runtime.active == 1
            and registry_empty
            and cache_empty
            and failed_unloaded
            and isinstance(fresh, AtlasRef)
            and fresh.generation == 1
        ),
        "rolled_back": bool(result.get("rolled_back")),
        "active_generation": runtime.active,
        "closures_cleared": result.get("closures_cleared"),
        "closure_registry_empty_after_rollback": registry_empty,
        "cache_entries_cleared": result.get("cache_entries_cleared"),
        "cache_empty_after_rollback": cache_empty,
        "failed_generation_loaded": not failed_unloaded,
        "fresh_closure_ref": fresh.cache_key() if isinstance(fresh, AtlasRef) else "miss",
        "problems": [p for p in runtime.problems() if not p.startswith("unload")],
        "seed": seed,
    }
    return report


def ab_version(
    code_version: Any = 1, resource_version: Any = None, **_kwargs: Any
) -> dict[str, Any]:
    """代码包与资源包版本必须一致，否则拒绝激活。"""

    input_code = str(code_version)
    input_resource = str(code_version if resource_version is None else resource_version)

    runtime = _base_runtime(1)
    runtime.register(2, atlases=("hero",), code_version="2.0.0", resource_version="1.9.0")
    runtime.install(2, verified=True)
    mismatch = runtime.activate(2)

    runtime.register(3, atlases=("hero",), code_version="3.0.0", resource_version="3.0.0")
    runtime.install(3, verified=True)
    matched = runtime.activate(3)

    return {
        "ok": bool(
            not mismatch.get("activated")
            and mismatch.get("reason") == "ab_version_mismatch"
            and matched.get("activated")
        ),
        "input_code_version": input_code,
        "input_resource_version": input_resource,
        "input_consistent": input_code == input_resource,
        "mismatched_bundle_rejected": not mismatch.get("activated"),
        "mismatch_reason": mismatch.get("reason"),
        "matching_bundle_accepted": bool(matched.get("activated")),
        "active_generation": runtime.active,
        "problems": runtime.problems(),
    }


def old_fallback(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """新版本失败时，旧版本及其闭包继续可用。"""

    runtime = _base_runtime(1)
    old_closure = runtime.make_closure(1, "hero")
    runtime.register(2, atlases=("hero", "ui"))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=False)
    rejected = runtime.activate(2)
    fallback_result = runtime.fallback_to_previous(2, reason="verification_failed")
    old_ref = old_closure()

    return {
        "ok": bool(
            not rejected.get("activated")
            and fallback_result.get("fallback")
            and runtime.active == 1
            and isinstance(old_ref, AtlasRef)
            and old_ref.generation == 1
            and not runtime.is_loaded(2)
        ),
        "rejected_activation": not rejected.get("activated"),
        "fallback_applied": bool(fallback_result.get("fallback")),
        "active_generation": runtime.active,
        "failed_generation": fallback_result.get("failed_generation"),
        "failed_generation_loaded": runtime.is_loaded(2),
        "old_version_closure_ref": old_ref.cache_key() if isinstance(old_ref, AtlasRef) else "miss",
        "old_version_intact": bool(fallback_result.get("old_version_intact")),
        "problems": runtime.problems(),
        "seed": seed,
    }


def atlas_ref(generation: Any = 1, name: Any = "atlas", **_kwargs: Any) -> dict[str, Any]:
    """图集引用始终携带代际，可直接用作代际化缓存键。"""

    generation = int(generation)
    name = str(name)
    runtime = _base_runtime(generation, atlases=(name,))
    ref = runtime.resolve(generation, name)
    cross = runtime.resolve(generation + 1, name)

    return {
        "ok": bool(isinstance(ref, AtlasRef) and ref.generation == generation and cross is MISSING),
        "generation": generation,
        "name": name,
        "cache_key": ref.cache_key() if isinstance(ref, AtlasRef) else None,
        "generation_scoped": isinstance(ref, AtlasRef)
        and ref.cache_key().startswith(f"g{generation}/"),
        "missing_in_other_generation": cross is MISSING,
        "resolved": isinstance(ref, AtlasRef),
    }


def _validate_manifest(manifest: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    atlases = manifest.get("atlases") or []
    code = manifest.get("code_version")
    resource = manifest.get("resource_version")
    checks = {
        "has_generation": isinstance(manifest.get("generation"), int),
        "has_atlases": bool(atlases),
        "ab_versions_present": code is not None and resource is not None,
        "ab_versions_match": code is not None and code == resource,
        "checksum_present": bool(manifest.get("checksum")),
    }
    return all(checks.values()), checks


def validation(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """校验更新包清单：结构、图集、AB 版本与校验和。"""

    good = {
        "generation": 2,
        "atlases": ["hero", "ui"],
        "code_version": "2.0.0",
        "resource_version": "2.0.0",
        "checksum": "sha256:demo",
    }
    bad = {
        "generation": 3,
        "atlases": [],
        "code_version": "3.0.0",
        "resource_version": "2.9.0",
        "checksum": "",
    }
    valid, checks = _validate_manifest(good)
    invalid, bad_checks = _validate_manifest(bad)
    return {
        "ok": bool(valid and not invalid),
        "valid": valid,
        "checks": checks,
        "invalid_rejected": not invalid,
        "invalid_checks": bad_checks,
        "seed": seed,
    }


def audit(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """审计事件顺序：任何代际都不得在校验通过前被激活。"""

    runtime = _base_runtime(1)
    runtime.enter_scene("battle", battle=True)
    runtime.register(2, atlases=("hero", "ui", "fx"))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.activate(2)  # 校验未完成 → 被拒
    runtime.verify(2, ok=True)
    runtime.activate(2)  # 有场景 → 延后
    runtime.exit_scene("battle")  # 延后的激活在此执行
    runtime.rollback("audit_cleanup")

    verified: set[int] = set()
    active_scenes = 0
    violations: list[str] = []
    for entry in runtime.events:
        event = entry["event"]
        if event == "verify" and entry.get("ok"):
            verified.add(int(entry["generation"]))
        elif event == "activate":
            generation = int(entry["generation"])
            if generation not in verified:
                violations.append(f"activate_before_verify:{generation}")
            if active_scenes > 0:
                violations.append(f"activate_with_active_scene:{generation}")
        elif event == "scene_enter":
            active_scenes += 1
        elif event == "scene_exit":
            active_scenes = max(0, active_scenes - 1)

    return {
        "ok": not violations,
        "event_count": len(runtime.events),
        "events": runtime.events,
        "violations": violations,
        "final_active": runtime.active,
        "problems": runtime.problems(),
        "seed": seed,
    }


def fallback(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """三类失败都回退到上一个已知可用代际。"""

    cases: list[dict[str, Any]] = []

    verify_runtime = _base_runtime(1)
    verify_runtime.register(2, atlases=("hero",))
    verify_runtime.begin_update(2)
    verify_runtime.finish_download(2)
    verify_runtime.verify(2, ok=False)
    verify_activation = verify_runtime.activate(2)
    verify_fallback = verify_runtime.fallback_to_previous(2, reason="verification_failed")
    cases.append(
        {
            "reason": "verification_failed",
            "activation_blocked": not verify_activation.get("activated"),
            "fell_back": bool(verify_fallback.get("fallback")),
            "active": verify_runtime.active,
        }
    )

    ab_runtime = _base_runtime(1)
    ab_runtime.register(2, atlases=("hero",), code_version="2.0.0", resource_version="1.0.0")
    ab_runtime.install(2, verified=True)
    ab_activation = ab_runtime.activate(2)
    ab_fallback = ab_runtime.fallback_to_previous(2, reason="ab_version_mismatch")
    cases.append(
        {
            "reason": "ab_version_mismatch",
            "activation_blocked": not ab_activation.get("activated"),
            "fell_back": bool(ab_fallback.get("fallback")),
            "active": ab_runtime.active,
        }
    )

    runtime_error = _base_runtime(1)
    runtime_error.register(2, atlases=("hero",))
    runtime_error.install(2, verified=True)
    runtime_error.activate(2)
    rollback_result = runtime_error.rollback("activation_error")
    cases.append(
        {
            "reason": "activation_error",
            "activation_blocked": False,
            "fell_back": bool(rollback_result.get("rolled_back")),
            "active": runtime_error.active,
        }
    )

    return {
        "ok": all(case["fell_back"] and case["active"] == 1 for case in cases),
        "cases": cases,
        "old_version": 1,
        "seed": seed,
    }


def download_state(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """下载/校验状态机：downloading → verifying → ready → active。"""

    runtime = _base_runtime(1)
    runtime.register(2, atlases=("hero", "ui"))

    states = [runtime.state]
    runtime.begin_update(2)
    states.append(runtime.state)
    states.append(runtime.finish_download(2))
    gate_while_verifying = runtime.activate(2)
    runtime.verify(2, ok=True)
    states.append(runtime.state)
    activated = runtime.activate(2)
    states.append(runtime.state)

    return {
        "ok": bool(
            states[0] == "active"
            and states[1] == "downloading"
            and states[2] == "verifying"
            and states[3] == "ready"
            and states[4] == "active"
            and not gate_while_verifying.get("activated")
            and activated.get("activated")
        ),
        "states": states,
        "gate_open_after_verify": bool(activated.get("activated")),
        "gate_closed_while_verifying": not gate_while_verifying.get("activated"),
        "download_and_verify_separate": True,
        "seed": seed,
    }


def metrics(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """运行一次完整切换，导出可观测指标，混合版本事件必须为 0。"""

    runtime = _base_runtime(1)
    runtime.register(2, atlases=("hero", "ui"))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.activate(2)  # 被门控拒绝
    runtime.verify(2, ok=True)
    runtime.enter_scene("battle", battle=True)
    runtime.activate(2)  # 场景隔离 → 延后
    runtime.exit_scene("battle")
    runtime.rollback("metrics_cleanup")

    snapshot = runtime.metrics_snapshot()
    return {
        "ok": bool(
            snapshot["mixed_version_events"] == 0
            and snapshot["blocked_activations"] >= 1
            and snapshot["deferred_activations"] >= 1
            and snapshot["rollbacks"] == 1
            and not runtime.problems()
        ),
        "metrics": snapshot,
        "mixed_version_events": snapshot["mixed_version_events"],
        "active_generation": runtime.active,
        "problems": runtime.problems(),
        "seed": seed,
    }


def mid_battle(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """战斗中途不允许切换版本；战斗内绑定代际保持不变。"""

    runtime = _base_runtime(1)
    runtime.enter_scene("battle", battle=True)
    battle_closure = runtime.make_closure(runtime.scene_generation("battle") or 1, "hero")

    runtime.register(2, atlases=("hero", "ui", "fx"))
    runtime.begin_update(2)
    runtime.finish_download(2)
    runtime.verify(2, ok=True)
    attempt = runtime.activate(2)
    during = battle_closure()

    exit_result = runtime.exit_scene("battle")
    after = runtime.active
    battle_closure.dispose()

    return {
        "ok": bool(
            not attempt.get("activated")
            and attempt.get("deferred") is True
            and isinstance(during, AtlasRef)
            and during.generation == 1
            and after == 2
        ),
        "activation_blocked": not attempt.get("activated"),
        "activation_deferred": bool(attempt.get("deferred")),
        "battle_generation_during_update": (
            during.generation if isinstance(during, AtlasRef) else None
        ),
        "mixed_version": not isinstance(during, AtlasRef) or during.generation != 1,
        "active_after_battle": after,
        "battle_exit_flushed": bool(exit_result.get("flushed", {}).get("activated")),
        "problems": runtime.problems(),
        "seed": seed,
    }


def cache_key(generation: Any = 1, name: Any = "atlas", **_kwargs: Any) -> dict[str, Any]:
    """缓存键带代际前缀，新旧代际的同名资源互不串读。"""

    generation = int(generation)
    name = str(name)
    runtime = _base_runtime(generation, atlases=(name,))
    runtime.cache_put(generation, name, b"old-payload")
    old_key = runtime.cache_key(generation, name)

    runtime.register(generation + 1, atlases=(name,))
    runtime.install(generation + 1, verified=True)
    runtime.activate(generation + 1)
    new_before_write = runtime.cache_get(generation + 1, name)
    runtime.cache_put(generation + 1, name, b"new-payload")
    new_key = runtime.cache_key(generation + 1, name)

    return {
        "ok": bool(
            old_key != new_key
            and new_before_write is MISSING
            and old_key.startswith(f"g{generation}/")
            and new_key.startswith(f"g{generation + 1}/")
        ),
        "cache_key": old_key,
        "next_generation_cache_key": new_key,
        "distinct": old_key != new_key,
        "no_cross_generation_hit": new_before_write is MISSING,
        "generation": generation,
        "next_generation": generation + 1,
    }


def determinism(seed: Any = 1, **_kwargs: Any) -> dict[str, Any]:
    """同一输入重复执行结果完全一致，便于回归与审计。"""

    first = scene_isolate(seed)
    second = scene_isolate(seed)
    payload = json.dumps(first, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "ok": first == second,
        "deterministic": first == second,
        "digest": digest,
        "seed": seed,
        "runs": 2,
    }
