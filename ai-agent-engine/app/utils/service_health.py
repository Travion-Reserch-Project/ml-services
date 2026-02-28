"""
Service Health Monitoring for Active Guardian Components.

This module provides comprehensive health monitoring and circuit breaker functionality
for all external services used by the Travion AI Engine. It tracks service availability,
degradation, and failures to prevent cascading issues and provide clear visibility.

Key Components:
    - ServiceStatus: Enum for service health states
    - ServiceType: Enum for all monitored services
    - ServiceHealth: Dataclass tracking health status per service
    - ServiceHealthMonitor: Global singleton tracking all services
    - CircuitBreaker: Prevents cascading failures by blocking failing services

Usage:
    from app.utils.service_health import get_health_monitor, ServiceType

    health_monitor = get_health_monitor()
    health_monitor.report_success(ServiceType.WEATHER_API)
    health_monitor.report_failure(ServiceType.NEWS_API, "API timeout")

    if health_monitor.is_service_available(ServiceType.WEATHER_API):
        # Proceed with weather API call
        pass
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Types of services monitored in the system."""
    WEATHER_API = "weather_api"
    NEWS_API = "news_api"
    EVENT_SENTINEL = "event_sentinel"
    CROWDCAST = "crowdcast"
    GOLDEN_HOUR = "golden_hour"
    LLM = "llm"
    CHROMADB = "chromadb"


@dataclass
class ServiceHealth:
    """
    Health status tracking for a service.

    Attributes:
        service_type: Type of service being monitored
        status: Current health status
        last_check: When health was last assessed
        error_message: Most recent error message if any
        consecutive_failures: Count of consecutive failures
        last_success: When service last succeeded
        metadata: Additional service-specific data
    """
    service_type: ServiceType
    status: ServiceStatus
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY

    def is_degraded(self) -> bool:
        """Check if service is degraded."""
        return self.status == ServiceStatus.DEGRADED

    def is_unavailable(self) -> bool:
        """Check if service is unavailable."""
        return self.status == ServiceStatus.UNAVAILABLE


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily blocking calls to failing services.
    After a threshold of failures, the circuit "opens" and blocks requests for a timeout
    period before attempting recovery.

    States:
        - CLOSED: Normal operation, requests allowed
        - OPEN: Too many failures, requests blocked
        - HALF_OPEN: Testing if service recovered (automatic after timeout)

    Args:
        service_name: Name of the service being protected
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Seconds to wait before attempting recovery
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._is_open = False

    def record_failure(self):
        """Record a service failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self._is_open = True
            logger.error(
                f"🔴 Circuit breaker OPENED for {self.service_name} "
                f"(failures: {self.failure_count}/{self.failure_threshold})"
            )

    def record_success(self):
        """Record a successful call, resetting the circuit."""
        if self._is_open:
            logger.info(f"🟢 Circuit breaker CLOSED for {self.service_name} (recovered)")
        self.failure_count = 0
        self._is_open = False

    def is_open(self) -> bool:
        """
        Check if circuit breaker is open (blocking calls).

        If timeout has passed since last failure, attempt recovery by closing circuit.
        """
        if not self._is_open:
            return False

        # Check if timeout has passed for recovery attempt
        if self.last_failure_time:
            time_since_failure = datetime.utcnow() - self.last_failure_time
            if time_since_failure >= self.timeout:
                logger.info(
                    f"🟡 Circuit breaker for {self.service_name} attempting recovery "
                    f"(timeout: {self.timeout.total_seconds()}s passed)"
                )
                # Transition to HALF_OPEN state (next call will test)
                self._is_open = False
                self.failure_count = 0
                return False

        return True

    def can_attempt(self) -> bool:
        """Check if we can attempt a service call."""
        return not self.is_open()


class ServiceHealthMonitor:
    """
    Global health monitor for all services.

    Tracks service availability and manages circuit breakers for resilience.
    This is a singleton - use get_health_monitor() to access.

    Example:
        monitor = get_health_monitor()
        monitor.register_service(ServiceType.WEATHER_API)

        # After successful API call
        monitor.report_success(ServiceType.WEATHER_API)

        # After failed API call
        monitor.report_failure(ServiceType.WEATHER_API, "Timeout")

        # Check before making call
        if monitor.is_service_available(ServiceType.WEATHER_API):
            # Safe to call
            pass
    """

    def __init__(self):
        self._health_status: Dict[ServiceType, ServiceHealth] = {}
        self._circuit_breakers: Dict[ServiceType, CircuitBreaker] = {}
        logger.info("🏥 Service Health Monitor initialized")

    def register_service(
        self,
        service_type: ServiceType,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Register a service for monitoring.

        Args:
            service_type: Type of service to monitor
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before recovery attempt
        """
        self._health_status[service_type] = ServiceHealth(
            service_type=service_type,
            status=ServiceStatus.UNKNOWN
        )
        self._circuit_breakers[service_type] = CircuitBreaker(
            service_name=service_type.value,
            failure_threshold=circuit_breaker_threshold,
            timeout_seconds=circuit_breaker_timeout
        )
        logger.debug(f"📋 Registered service for monitoring: {service_type.value}")

    def report_success(
        self,
        service_type: ServiceType,
        metadata: Optional[Dict] = None
    ):
        """
        Report successful service call.

        Args:
            service_type: Service that succeeded
            metadata: Optional additional context
        """
        if service_type not in self._health_status:
            self.register_service(service_type)

        health = self._health_status[service_type]
        health.status = ServiceStatus.HEALTHY
        health.last_check = datetime.utcnow()
        health.last_success = datetime.utcnow()
        health.consecutive_failures = 0
        health.error_message = None
        if metadata:
            health.metadata.update(metadata)

        # Reset circuit breaker
        if service_type in self._circuit_breakers:
            self._circuit_breakers[service_type].record_success()

        logger.debug(f"✅ {service_type.value}: SUCCESS")

    def report_failure(
        self,
        service_type: ServiceType,
        error_message: str,
        metadata: Optional[Dict] = None
    ):
        """
        Report service call failure.

        Args:
            service_type: Service that failed
            error_message: Error description
            metadata: Optional additional context
        """
        if service_type not in self._health_status:
            self.register_service(service_type)

        health = self._health_status[service_type]
        health.last_check = datetime.utcnow()
        health.consecutive_failures += 1
        health.error_message = error_message
        if metadata:
            health.metadata.update(metadata)

        # Update circuit breaker
        circuit_breaker = self._circuit_breakers.get(service_type)
        if circuit_breaker:
            circuit_breaker.record_failure()

            if circuit_breaker.is_open():
                health.status = ServiceStatus.UNAVAILABLE
                logger.error(
                    f"❌ {service_type.value}: UNAVAILABLE after "
                    f"{health.consecutive_failures} failures - {error_message}"
                )
            elif health.consecutive_failures >= 3:
                health.status = ServiceStatus.DEGRADED
                logger.warning(
                    f"⚠️ {service_type.value}: DEGRADED "
                    f"({health.consecutive_failures} failures) - {error_message}"
                )
            else:
                health.status = ServiceStatus.DEGRADED
                logger.warning(f"⚠️ {service_type.value}: FAILURE - {error_message}")

    def get_health(self, service_type: ServiceType) -> ServiceHealth:
        """
        Get health status for a service.

        Args:
            service_type: Service to check

        Returns:
            ServiceHealth object for the service
        """
        if service_type not in self._health_status:
            self.register_service(service_type)
        return self._health_status[service_type]

    def get_all_health(self) -> Dict[ServiceType, ServiceHealth]:
        """Get health status for all registered services."""
        return self._health_status.copy()

    def is_service_available(self, service_type: ServiceType) -> bool:
        """
        Check if service is available for use.

        Args:
            service_type: Service to check

        Returns:
            True if service can be used, False if circuit is open
        """
        health = self.get_health(service_type)
        circuit_breaker = self._circuit_breakers.get(service_type)

        # Service unavailable if circuit is open
        if circuit_breaker and circuit_breaker.is_open():
            return False

        # Service unavailable if status is UNAVAILABLE
        return health.status != ServiceStatus.UNAVAILABLE

    def get_circuit_breaker(self, service_type: ServiceType) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service."""
        return self._circuit_breakers.get(service_type)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all service health statuses.

        Returns:
            Dictionary with health summary statistics
        """
        all_health = self.get_all_health()

        healthy = [s for s in all_health.values() if s.is_healthy()]
        degraded = [s for s in all_health.values() if s.is_degraded()]
        unavailable = [s for s in all_health.values() if s.is_unavailable()]

        return {
            "total_services": len(all_health),
            "healthy_count": len(healthy),
            "degraded_count": len(degraded),
            "unavailable_count": len(unavailable),
            "overall_status": (
                "healthy" if len(unavailable) == 0 and len(degraded) == 0
                else "degraded" if len(unavailable) == 0
                else "unhealthy"
            )
        }


# Global service health monitor (singleton)
_health_monitor: Optional[ServiceHealthMonitor] = None


def get_health_monitor() -> ServiceHealthMonitor:
    """
    Get or create the global health monitor singleton.

    Returns:
        The global ServiceHealthMonitor instance
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ServiceHealthMonitor()
    return _health_monitor
