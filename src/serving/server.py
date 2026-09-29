"""The serving layer: pick a version, score, measure, and roll back if it goes wrong.

Shadow traffic is the piece most platforms omit and the one that makes a deployment
safe to reason about. A shadow model scores the same request as the champion, its
output is recorded and never returned, and **its failures cannot affect the response**.
That last property is the whole point: if a shadow error could surface to a caller,
you would not dare shadow anything worth testing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .observe.slo import SLOMonitor
from .registry.store import Registry, RegistryError
from .routing.split import in_canary
from .types import Prediction, Stage

# A loaded model: takes features, returns a prediction.
ModelFn = Callable[[dict[str, Any]], Any]


class ServingError(RuntimeError):
    pass


@dataclass
class ServingPlatform:
    registry: Registry = field(default_factory=Registry)
    monitor: SLOMonitor = field(default_factory=SLOMonitor)
    # key "name:vN" -> callable
    loaded: dict[str, ModelFn] = field(default_factory=dict)
    auto_rollback: bool = True
    rollbacks: list[dict] = field(default_factory=list)

    def load(self, name: str, version: int, fn: ModelFn) -> None:
        self.loaded[f"{name}:v{version}"] = fn

    def _call(self, key: str, features: dict) -> tuple[Any, float, str]:
        fn = self.loaded.get(key)
        if fn is None:
            raise ServingError(f"{key} is registered but not loaded")
        started = time.perf_counter()
        try:
            output = fn(features)
            return output, (time.perf_counter() - started) * 1000, ""
        except Exception as exc:
            return None, (time.perf_counter() - started) * 1000, str(exc)

    def predict(self, name: str, features: dict, *, request_key: str | None = None) -> Prediction:
        # `or` would treat "" as missing and assign a fresh random key on every call,
        # silently breaking stickiness for a caller using "" as a legitimate (if
        # unusual) anonymous/guest identifier. Only None means "no key given".
        if request_key is None:
            request_key = uuid.uuid4().hex
        champion = self.registry.champion(name)
        if champion is None:
            raise ServingError(f"{name} has no champion")

        challenger = self.registry.challenger(name)
        serving = champion
        stage = Stage.CHAMPION
        if challenger is not None and in_canary(request_key, challenger.traffic, salt=name):
            serving, stage = challenger, Stage.CHALLENGER

        output, latency, error = self._call(serving.key, features)
        self.monitor.record(serving.key, latency, error=bool(error))

        prediction = Prediction(
            model=name,
            version=serving.version,
            output=output,
            latency_ms=round(latency, 3),
            stage=stage,
            request_id=request_key,
            error=error,
        )

        # Shadows score the same request. Never returned, never able to fail the call.
        shadows = self.registry.shadows(name)
        if shadows:
            recorded = {}
            for shadow in shadows:
                try:
                    s_out, s_latency, s_error = self._call(shadow.key, features)
                except Exception as exc:  # a shadow must not be able to break serving
                    s_out, s_latency, s_error = None, 0.0, str(exc)
                self.monitor.record(shadow.key, s_latency, error=bool(s_error))
                recorded[shadow.key] = {
                    "output": s_out,
                    "latency_ms": round(s_latency, 3),
                    "error": s_error,
                    "agrees_with_champion": s_out == output and not s_error and not error,
                }
            prediction.shadow = recorded

        if self.auto_rollback and challenger is not None:
            self._maybe_rollback(name, challenger.key)

        return prediction

    def _maybe_rollback(self, name: str, challenger_key: str) -> None:
        reason = self.monitor.breached(challenger_key)
        if reason is None:
            return
        # Only roll back if the champion is *not* equally broken. A shared upstream
        # outage degrades both, and rolling back then removes a healthy deployment
        # while fixing nothing.
        champion = self.registry.champion(name)
        if champion is not None:
            if self.monitor.breached(champion.key) is not None:
                return
            # `breached()` is silent below min_samples, so None means either "healthy"
            # or "no verdict yet" - and treating the second as the first is how an
            # upstream outage gets misread as a bad canary. The challenger can reach
            # min_samples first (hash bucketing is not exactly even, and a large canary
            # share makes it likely), at which point the champion is failing every
            # request and still has nothing to say about it. Wait for a verdict.
            if self.monitor.get(champion.key).requests < self.monitor.slo.min_samples:
                return
        try:
            self.registry.rollback(name, reason=f"SLO breach: {reason}")
        except RegistryError:
            return
        self.rollbacks.append({"model": name, "version": challenger_key, "reason": reason})

    def status(self, name: str) -> dict:
        champion = self.registry.champion(name)
        challenger = self.registry.challenger(name)
        out: dict = {
            "model": name,
            "champion": champion.version if champion else None,
            "challenger": challenger.version if challenger else None,
            "canary_traffic": challenger.traffic if challenger else 0.0,
            "shadows": [s.version for s in self.registry.shadows(name)],
            "rollbacks": len(self.rollbacks),
        }
        if champion and challenger:
            out["comparison"] = self.monitor.compare(champion.key, challenger.key)
        elif champion:
            out["champion_stats"] = self.monitor.get(champion.key).summary()
        return out
