"""Serving platform tests.

Models are fakes: a function that returns a value, or fails, or is slow. Everything
worth testing here is a *routing and lifecycle* behaviour, not a modelling one, so the
whole platform is verifiable with no training, no weights and no GPU.
"""

from __future__ import annotations

import time

import pytest

from serving.observe.slo import SLO, SLOMonitor
from serving.registry.store import Registry, RegistryError
from serving.routing.split import bucket, in_canary
from serving.server import ServingError, ServingPlatform
from serving.types import Stage


def good(value=1):
    return lambda features: value


def broken(features):
    raise RuntimeError("model exploded")


def slow(ms=50):
    def fn(features):
        time.sleep(ms / 1000)
        return 1

    return fn


def platform(**kw) -> ServingPlatform:
    p = ServingPlatform(**kw)
    p.registry.register("risk", 1)
    p.registry.register("risk", 2)
    p.load("risk", 1, good(1))
    p.load("risk", 2, good(2))
    p.registry.promote("risk", 1)
    return p


class TestRegistry:
    def test_one_champion_at_a_time(self):
        """A registry that can briefly have two champions eventually will."""
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        r.promote("m", 2)
        champions = [v for v in r.versions("m") if v.stage is Stage.CHAMPION]
        assert len(champions) == 1 and champions[0].version == 2

    def test_promotion_is_idempotent(self):
        r = Registry()
        r.register("m", 1)
        r.promote("m", 1)
        r.promote("m", 1)
        assert len([h for h in r.history if h["action"] == "promote"]) == 1

    def test_duplicate_version_is_refused(self):
        r = Registry()
        r.register("m", 1)
        with pytest.raises(RegistryError):
            r.register("m", 1)

    def test_canary_without_a_champion_is_refused(self):
        """A canary is defined relative to what it is compared against."""
        r = Registry()
        r.register("m", 1)
        with pytest.raises(RegistryError):
            r.start_canary("m", 1, 0.1)

    def test_only_one_canary_at_a_time(self):
        """Two simultaneous canaries make the attribution meaningless."""
        r = Registry()
        for v in (1, 2, 3):
            r.register("m", v)
        r.promote("m", 1)
        r.start_canary("m", 2, 0.1)
        with pytest.raises(RegistryError):
            r.start_canary("m", 3, 0.1)

    def test_canary_traffic_must_be_a_proper_fraction(self):
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        with pytest.raises(RegistryError):
            r.start_canary("m", 2, 1.0)

    def test_rollback_archives_rather_than_demoting_to_shadow(self):
        """After a failure the question is settled; leaving it scoring traffic
        invites someone to promote it by mistake."""
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        r.start_canary("m", 2, 0.5)
        r.rollback("m", reason="bad")
        assert r.get("m", 2).stage is Stage.ARCHIVED
        assert r.challenger("m") is None

    def test_history_records_every_transition(self):
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        r.start_canary("m", 2, 0.5)
        r.rollback("m")
        assert [h["action"] for h in r.history] == ["promote", "canary", "rollback"]

    def test_add_shadow_cannot_steal_the_champion(self):
        """Shadowing the champion in place would silently leave the model with no
        champion at all - the next predict() fails far from this call site."""
        r = Registry()
        r.register("m", 1)
        r.promote("m", 1)
        with pytest.raises(RegistryError):
            r.add_shadow("m", 1)
        assert r.champion("m") is not None

    def test_add_shadow_cannot_steal_the_challenger(self):
        """Shadowing the challenger in place would silently stop it receiving
        canary traffic while the registry still calls it the challenger."""
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        r.start_canary("m", 2, 0.1)
        with pytest.raises(RegistryError):
            r.add_shadow("m", 2)
        assert r.challenger("m") is not None
        assert r.challenger("m").version == 2

    def test_add_shadow_on_an_unrelated_version_still_works(self):
        r = Registry()
        r.register("m", 1)
        r.register("m", 2)
        r.promote("m", 1)
        r.add_shadow("m", 2)
        assert r.get("m", 2).stage is Stage.SHADOW
        assert "shadow" in [h["action"] for h in r.history]


class TestTrafficSplit:
    def test_assignment_is_sticky(self):
        """Random assignment lets one user see both versions in a session, which
        produces an A/B result that measures nothing."""
        assert all(in_canary("user-42", 0.5) == in_canary("user-42", 0.5) for _ in range(100))

    def test_split_is_roughly_the_requested_fraction(self):
        hits = sum(in_canary(f"user-{i}", 0.2) for i in range(10_000))
        assert 0.18 < hits / 10_000 < 0.22

    def test_zero_and_one_are_absolute(self):
        assert not in_canary("u", 0.0)
        assert in_canary("u", 1.0)

    def test_salt_decorrelates_experiments(self):
        """A user unlucky in one experiment should not be unlucky in every one."""
        a = [in_canary(f"u{i}", 0.5, salt="exp-a") for i in range(2000)]
        b = [in_canary(f"u{i}", 0.5, salt="exp-b") for i in range(2000)]
        agreement = sum(x == y for x, y in zip(a, b, strict=False)) / len(a)
        assert 0.45 < agreement < 0.55

    def test_bucket_is_in_range(self):
        assert all(0.0 <= bucket(f"k{i}") < 1.0 for i in range(1000))


class TestServing:
    def test_all_traffic_goes_to_the_champion_by_default(self):
        p = platform()
        assert all(p.predict("risk", {}).version == 1 for _ in range(50))

    def test_canary_receives_roughly_its_share(self):
        p = platform()
        p.registry.start_canary("risk", 2, 0.3)
        hits = sum(p.predict("risk", {}, request_key=f"u{i}").version == 2 for i in range(2000))
        assert 0.27 < hits / 2000 < 0.33

    def test_serving_without_a_champion_fails_loudly(self):
        p = ServingPlatform()
        p.registry.register("risk", 1)
        with pytest.raises(ServingError):
            p.predict("risk", {})

    def test_a_registered_but_unloaded_model_is_an_error_not_a_crash(self):
        p = platform()
        p.registry.register("risk", 3)
        p.registry.promote("risk", 3)
        with pytest.raises(ServingError):
            p.predict("risk", {})

    def test_a_failing_model_is_recorded_not_raised(self):
        p = platform(auto_rollback=False)
        p.load("risk", 1, broken)
        result = p.predict("risk", {})
        assert result.error and result.output is None

    def test_empty_string_request_key_is_sticky_like_any_other_key(self):
        """ "" is a legitimate anonymous/guest key. `key or default` treats it as
        missing and would hand out a fresh random key every call, silently
        destroying stickiness for exactly the caller who has no better key."""
        p = platform()
        p.registry.start_canary("risk", 2, 0.5)
        results = [p.predict("risk", {}, request_key="").version for _ in range(20)]
        assert len(set(results)) == 1
        assert results[0] == p.predict("risk", {}, request_key="").version

    def test_none_request_key_still_gets_a_fresh_random_key(self):
        """Only the *absence* of a key (None, or the parameter left out) should
        fall back to a random one - not an empty string."""
        p = platform()
        first = p.predict("risk", {}).request_id
        second = p.predict("risk", {}).request_id
        assert first != second


class TestShadow:
    def test_shadow_output_is_recorded_but_never_returned(self):
        p = platform()
        p.registry.add_shadow("risk", 2)
        result = p.predict("risk", {})
        assert result.output == 1  # champion's answer
        assert result.shadow["risk:v2"]["output"] == 2

    def test_a_failing_shadow_cannot_break_the_response(self):
        """If a shadow failure could surface to a caller, nobody would dare shadow
        anything worth testing."""
        p = platform()
        p.load("risk", 2, broken)
        p.registry.add_shadow("risk", 2)
        result = p.predict("risk", {})
        assert result.output == 1 and not result.error
        assert result.shadow["risk:v2"]["error"]

    def test_agreement_is_reported(self):
        p = platform()
        p.load("risk", 2, good(1))  # same answer as champion
        p.registry.add_shadow("risk", 2)
        assert p.predict("risk", {}).shadow["risk:v2"]["agrees_with_champion"]


class TestSLO:
    def test_no_verdict_below_the_sample_floor(self):
        """One slow request in the first three must not roll back a healthy deploy."""
        m = SLOMonitor(slo=SLO(max_p95_ms=10, min_samples=50))
        for _ in range(5):
            m.record("m:v1", 5000)
        assert m.breached("m:v1") is None

    def test_latency_breach_is_detected(self):
        m = SLOMonitor(slo=SLO(max_p95_ms=100, min_samples=10))
        for _ in range(50):
            m.record("m:v1", 500)
        assert "p95" in m.breached("m:v1")

    def test_error_rate_breach_is_detected(self):
        m = SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=10))
        for i in range(100):
            m.record("m:v1", 10, error=i % 4 == 0)
        assert "error rate" in m.breached("m:v1")

    def test_percentiles_are_not_the_mean(self):
        """The mean is the one latency statistic that never matters."""
        m = SLOMonitor()
        for _ in range(99):
            m.record("m:v1", 10)
        m.record("m:v1", 10_000)
        stats = m.get("m:v1")
        assert stats.percentile(50) == 10
        assert stats.percentile(99) >= 10

    def test_comparison_is_relative(self):
        """A challenger at 3% errors is fine against a champion at 4%."""
        m = SLOMonitor()
        for i in range(100):
            m.record("m:v1", 100, error=i % 25 == 0)
            m.record("m:v2", 90, error=i % 34 == 0)
        c = m.compare("m:v1", "m:v2")
        assert c["error_rate_delta"] < 0
        assert c["p95_delta_ms"] < 0


class TestAutoRollback:
    def test_a_bad_canary_rolls_itself_back(self):
        p = platform(monitor=SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=20)))
        p.load("risk", 2, broken)
        p.registry.start_canary("risk", 2, 0.9)
        for i in range(200):
            p.predict("risk", {}, request_key=f"u{i}")
        assert p.registry.challenger("risk") is None
        assert p.rollbacks and "error rate" in p.rollbacks[0]["reason"]

    def test_a_healthy_canary_survives(self):
        p = platform(monitor=SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=20)))
        p.registry.start_canary("risk", 2, 0.5)
        for i in range(300):
            p.predict("risk", {}, request_key=f"u{i}")
        assert p.registry.challenger("risk") is not None
        assert not p.rollbacks

    def test_a_shared_outage_does_not_roll_back_the_canary(self):
        """Both versions degrading means the problem is upstream. Rolling back then
        removes a healthy deployment and fixes nothing."""
        p = platform(monitor=SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=20)))
        p.load("risk", 1, broken)
        p.load("risk", 2, broken)
        p.registry.start_canary("risk", 2, 0.5)
        for i in range(300):
            p.predict("risk", {}, request_key=f"u{i}")
        assert p.registry.challenger("risk") is not None
        assert not p.rollbacks

    def test_a_shared_outage_is_survived_even_when_the_canary_is_judged_first(self):
        """The same guarantee as above, with request keys that bucket the other way.

        `breached()` is silent below min_samples, so it returns None both for "healthy"
        and for "no verdict yet". Whether the champion or the challenger crosses
        min_samples first is decided by the hash of the request keys - with `u{i}` keys
        the champion gets there first and the outage guard works, which is why the test
        above passed while this path was broken. With `req-{i}` keys the challenger is
        judged first, the champion is still silent, and the outage used to be misread
        as a bad canary.
        """
        p = platform(monitor=SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=20)))
        p.load("risk", 1, broken)
        p.load("risk", 2, broken)
        p.registry.start_canary("risk", 2, 0.5)
        for i in range(300):
            p.predict("risk", {}, request_key=f"req-{i}")
        assert p.registry.challenger("risk") is not None
        assert not p.rollbacks

    def test_auto_rollback_can_be_disabled(self):
        p = platform(
            auto_rollback=False, monitor=SLOMonitor(slo=SLO(max_error_rate=0.05, min_samples=20))
        )
        p.load("risk", 2, broken)
        p.registry.start_canary("risk", 2, 0.9)
        for i in range(200):
            p.predict("risk", {}, request_key=f"u{i}")
        assert p.registry.challenger("risk") is not None


class TestStatus:
    def test_reports_the_deployment_shape(self):
        p = platform()
        p.registry.start_canary("risk", 2, 0.25)
        p.predict("risk", {})
        s = p.status("risk")
        assert s["champion"] == 1 and s["challenger"] == 2
        assert s["canary_traffic"] == 0.25
        assert "comparison" in s
