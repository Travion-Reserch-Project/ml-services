"""
Tests for service health monitoring and circuit breaker functionality.

This test module validates:
- Circuit breaker opens after threshold failures
- Health monitor tracks service failures and successes
- Health monitor resets on success
- Service availability checks work correctly
"""

import pytest
from datetime import datetime, timedelta
from app.utils.service_health import (
    ServiceHealthMonitor,
    ServiceType,
    ServiceStatus,
    CircuitBreaker,
    get_health_monitor
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_opens_after_threshold(self):
        """Test that circuit breaker opens after threshold failures."""
        cb = CircuitBreaker("test_service", failure_threshold=3, timeout_seconds=60)

        assert cb.can_attempt() is True
        assert cb.is_open() is False

        # Record failures up to threshold
        for i in range(3):
            cb.record_failure()

        # Circuit should be open now
        assert cb.is_open() is True
        assert cb.can_attempt() is False

    def test_circuit_breaker_resets_on_success(self):
        """Test that circuit breaker resets after success."""
        cb = CircuitBreaker("test_service", failure_threshold=3, timeout_seconds=60)

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        # Record success
        cb.record_success()

        # Circuit should be closed and reset
        assert cb.is_open() is False
        assert cb.failure_count == 0
        assert cb.can_attempt() is True

    def test_circuit_breaker_recovers_after_timeout(self):
        """Test that circuit breaker attempts recovery after timeout."""
        cb = CircuitBreaker("test_service", failure_threshold=2, timeout_seconds=1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is True

        # Wait for timeout (1 second + small buffer)
        import time
        time.sleep(1.1)

        # Circuit should attempt recovery
        assert cb.is_open() is False
        assert cb.can_attempt() is True


class TestServiceHealthMonitor:
    """Test service health monitoring functionality."""

    def test_health_monitor_tracks_failures(self):
        """Test that health monitor tracks service failures."""
        monitor = ServiceHealthMonitor()
        monitor.register_service(ServiceType.WEATHER_API)

        # Report failure
        monitor.report_failure(
            ServiceType.WEATHER_API,
            "API key not configured"
        )

        health = monitor.get_health(ServiceType.WEATHER_API)
        assert health.status == ServiceStatus.DEGRADED
        assert health.consecutive_failures == 1
        assert "API key" in health.error_message

    def test_health_monitor_resets_on_success(self):
        """Test that health monitor resets after success."""
        monitor = ServiceHealthMonitor()
        monitor.register_service(ServiceType.WEATHER_API)

        # Report failures
        for _ in range(3):
            monitor.report_failure(ServiceType.WEATHER_API, "Temporary failure")

        health = monitor.get_health(ServiceType.WEATHER_API)
        assert health.consecutive_failures == 3
        assert health.status == ServiceStatus.DEGRADED

        # Report success
        monitor.report_success(ServiceType.WEATHER_API)

        health = monitor.get_health(ServiceType.WEATHER_API)
        assert health.status == ServiceStatus.HEALTHY
        assert health.consecutive_failures == 0
        assert health.error_message is None

    def test_health_monitor_opens_circuit_after_threshold(self):
        """Test that health monitor opens circuit breaker after threshold."""
        monitor = ServiceHealthMonitor()
        monitor.register_service(
            ServiceType.WEATHER_API,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=60
        )

        # Report failures up to threshold
        for i in range(5):
            monitor.report_failure(ServiceType.WEATHER_API, f"Failure {i+1}")

        health = monitor.get_health(ServiceType.WEATHER_API)
        circuit_breaker = monitor.get_circuit_breaker(ServiceType.WEATHER_API)

        # Circuit should be open
        assert circuit_breaker.is_open() is True
        assert health.status == ServiceStatus.UNAVAILABLE
        assert monitor.is_service_available(ServiceType.WEATHER_API) is False

    def test_health_monitor_summary(self):
        """Test that health monitor provides correct summary."""
        monitor = ServiceHealthMonitor()

        # Register and set different statuses
        monitor.register_service(ServiceType.WEATHER_API)
        monitor.register_service(ServiceType.NEWS_API)
        monitor.register_service(ServiceType.EVENT_SENTINEL)

        # Weather: healthy
        monitor.report_success(ServiceType.WEATHER_API)

        # News: degraded
        monitor.report_failure(ServiceType.NEWS_API, "Rate limit")

        # Event Sentinel: unavailable (open circuit)
        for _ in range(6):
            monitor.report_failure(ServiceType.EVENT_SENTINEL, "Service down")

        summary = monitor.get_summary()

        assert summary["total_services"] == 3
        assert summary["healthy_count"] == 1
        assert summary["degraded_count"] >= 1  # At least NEWS_API
        assert summary["unavailable_count"] >= 1  # EVENT_SENTINEL
        assert summary["overall_status"] in ["degraded", "unhealthy"]

    def test_health_monitor_metadata(self):
        """Test that health monitor tracks metadata correctly."""
        monitor = ServiceHealthMonitor()
        monitor.register_service(ServiceType.WEATHER_API)

        # Report success with metadata
        monitor.report_success(
            ServiceType.WEATHER_API,
            metadata={"api_version": "2.5", "endpoint": "weather"}
        )

        health = monitor.get_health(ServiceType.WEATHER_API)
        assert "api_version" in health.metadata
        assert health.metadata["api_version"] == "2.5"

        # Report failure with metadata
        monitor.report_failure(
            ServiceType.WEATHER_API,
            "Timeout",
            metadata={"response_time_ms": 5000}
        )

        health = monitor.get_health(ServiceType.WEATHER_API)
        assert "response_time_ms" in health.metadata
        assert health.metadata["response_time_ms"] == 5000


class TestServiceHealthSingleton:
    """Test that get_health_monitor() returns singleton."""

    def test_singleton_pattern(self):
        """Test that get_health_monitor returns the same instance."""
        monitor1 = get_health_monitor()
        monitor2 = get_health_monitor()

        assert monitor1 is monitor2

        # Modify state in one
        monitor1.register_service(ServiceType.WEATHER_API)
        monitor1.report_success(ServiceType.WEATHER_API)

        # Should be reflected in the other
        health = monitor2.get_health(ServiceType.WEATHER_API)
        assert health.is_healthy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
