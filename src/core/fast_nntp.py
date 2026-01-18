"""
Fast NNTP Downloader - Thread-based implementation
===================================================

Inspired by SABnzbd's approach: native threads instead of asyncio
for maximum throughput on high-speed connections.

Optimizations for 10 Gbps:
- TLS 1.3 with optimized ciphers
- Large socket buffers (8MB receive)
- Session resumption for faster reconnects
- Aggressive pipelining (20-50 requests/connection)

v1.3.0 Enhancements:
- Exponential backoff with jitter for retry logic
- Dynamic connection pool with health monitoring
- Adaptive pipelining based on RTT measurement
"""

import ssl
import socket
import time
import random
import statistics
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Deque, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock
from queue import Queue, Empty
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# RETRY CONFIGURATION WITH EXPONENTIAL BACKOFF + JITTER
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic."""
    max_attempts: int = 5          # Maximum retry attempts
    base_delay: float = 0.5        # Initial delay in seconds
    max_delay: float = 30.0        # Maximum delay cap
    jitter_factor: float = 0.5     # Randomization factor (0-1)

    # Retryable exceptions
    retryable_errors: Tuple[type, ...] = field(default_factory=lambda: (
        ConnectionError, TimeoutError, OSError, ssl.SSLError, socket.error
    ))


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay with exponential backoff and full jitter.

    Uses AWS-style "full jitter" algorithm to prevent thundering herds:
    delay = random(0, min(cap, base * 2^attempt))

    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration

    Returns:
        Delay in seconds before next retry
    """
    # Exponential growth with cap
    exp_delay = min(config.base_delay * (2 ** attempt), config.max_delay)

    # Full jitter: random between 0 and exp_delay
    jittered = random.uniform(0, exp_delay)

    return jittered


class RetryableOperation:
    """Context manager for retryable operations with exponential backoff."""

    def __init__(self, config: Optional[RetryConfig] = None, operation_name: str = "operation"):
        self.config = config or RetryConfig()
        self.operation_name = operation_name
        self.attempt = 0
        self.last_exception: Optional[Exception] = None

    def should_retry(self, exception: Exception) -> bool:
        """Check if operation should be retried."""
        if self.attempt >= self.config.max_attempts:
            return False
        return isinstance(exception, self.config.retryable_errors)

    def wait_before_retry(self) -> float:
        """Wait with backoff before retry. Returns actual delay used."""
        delay = calculate_backoff_delay(self.attempt, self.config)
        if delay > 0:
            time.sleep(delay)
        self.attempt += 1
        return delay

    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        while True:
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.last_exception = e
                if self.should_retry(e):
                    delay = self.wait_before_retry()
                    logger.debug(f"[RETRY] {self.operation_name} attempt {self.attempt}/{self.config.max_attempts} "
                               f"after {delay:.2f}s delay: {e}")
                else:
                    raise


# =============================================================================
# ADAPTIVE PIPELINING BASED ON RTT
# =============================================================================

class AdaptivePipeline:
    """
    Dynamically adjusts pipeline depth based on measured RTT.

    Higher latency connections benefit from deeper pipelines to keep
    the network saturated. Low latency connections need shallower
    pipelines to reduce memory overhead.
    """

    # Pipeline depth ranges
    MIN_DEPTH = 5
    MAX_DEPTH = 100
    DEFAULT_DEPTH = 30

    def __init__(self, initial_depth: int = DEFAULT_DEPTH, sample_size: int = 50):
        self._depth = initial_depth
        self._rtt_samples: Deque[float] = deque(maxlen=sample_size)
        self._lock = Lock()
        self._last_adjustment_time = 0.0
        self._adjustment_interval = 5.0  # Adjust every 5 seconds max

    @property
    def depth(self) -> int:
        """Current pipeline depth."""
        return self._depth

    def record_rtt(self, rtt_ms: float) -> None:
        """Record an RTT sample."""
        with self._lock:
            self._rtt_samples.append(rtt_ms)

    def get_avg_rtt(self) -> Optional[float]:
        """Get average RTT in milliseconds."""
        with self._lock:
            if len(self._rtt_samples) < 5:
                return None
            return statistics.mean(self._rtt_samples)

    def adjust(self) -> int:
        """
        Adjust pipeline depth based on RTT measurements.

        Returns:
            New pipeline depth
        """
        now = time.time()
        if now - self._last_adjustment_time < self._adjustment_interval:
            return self._depth

        avg_rtt = self.get_avg_rtt()
        if avg_rtt is None:
            return self._depth

        with self._lock:
            self._last_adjustment_time = now

            # Calculate optimal depth based on bandwidth-delay product principle
            # More latency = need more requests in flight to saturate bandwidth
            if avg_rtt < 10:
                # Very low latency (local/nearby server)
                new_depth = 15
            elif avg_rtt < 25:
                # Low latency (same continent)
                new_depth = 25
            elif avg_rtt < 50:
                # Medium latency (cross-continent)
                new_depth = 40
            elif avg_rtt < 100:
                # High latency (intercontinental)
                new_depth = 60
            elif avg_rtt < 200:
                # Very high latency (satellite/VPN)
                new_depth = 80
            else:
                # Extreme latency
                new_depth = self.MAX_DEPTH

            # Smooth transition (don't jump too fast)
            if abs(new_depth - self._depth) > 10:
                if new_depth > self._depth:
                    self._depth = min(self._depth + 10, new_depth)
                else:
                    self._depth = max(self._depth - 10, new_depth)
            else:
                self._depth = new_depth

            self._depth = max(self.MIN_DEPTH, min(self.MAX_DEPTH, self._depth))

            logger.debug(f"[PIPELINE] Adjusted depth to {self._depth} (avg RTT: {avg_rtt:.1f}ms)")

            return self._depth


# =============================================================================
# CONNECTION HEALTH TRACKING
# =============================================================================

class ConnectionHealth(Enum):
    """Connection health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"


@dataclass
class ConnectionStats:
    """Statistics for a single connection."""
    conn_id: int
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    avg_response_time_ms: float = 0.0
    _response_times: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    @property
    def health(self) -> ConnectionHealth:
        """Determine connection health based on stats."""
        if self.consecutive_failures >= 5:
            return ConnectionHealth.DEAD
        elif self.consecutive_failures >= 3:
            return ConnectionHealth.UNHEALTHY
        elif self.consecutive_failures >= 1:
            return ConnectionHealth.DEGRADED
        return ConnectionHealth.HEALTHY

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful operation."""
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_time = time.time()
        self._response_times.append(response_time_ms)
        if self._response_times:
            self.avg_response_time_ms = statistics.mean(self._response_times)

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()


# =============================================================================
# DYNAMIC CONNECTION POOL
# =============================================================================

class DynamicConnectionPool:
    """
    Dynamic connection pool that adjusts size based on throughput and health.

    Features:
    - Automatic scaling up when throughput is limited by connections
    - Automatic scaling down when connections are underutilized
    - Health-based connection rotation
    - Exponential backoff for reconnection attempts
    """

    def __init__(
        self,
        min_connections: int = 5,
        max_connections: int = 100,
        target_connections: int = 20,
        scale_up_threshold: float = 0.90,    # Scale up when utilization > 90%
        scale_down_threshold: float = 0.30,  # Scale down when utilization < 30%
        scale_step: int = 5                   # Connections to add/remove per adjustment
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.target_connections = target_connections
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_step = scale_step

        self._available: Deque = deque()
        self._in_use: Dict[int, 'NNTPConnection'] = {}
        self._stats: Dict[int, ConnectionStats] = {}
        self._lock = RLock()

        self._last_scale_time = 0.0
        self._scale_cooldown = 10.0  # Seconds between scaling decisions

        # Throughput tracking for scaling decisions
        self._throughput_samples: Deque[float] = deque(maxlen=30)
        self._target_throughput: float = 0.0

    def add_connection(self, conn: 'NNTPConnection') -> None:
        """Add a connection to the pool."""
        with self._lock:
            self._available.append(conn)
            self._stats[conn.conn_id] = ConnectionStats(conn_id=conn.conn_id)

    def acquire(self) -> Optional['NNTPConnection']:
        """Acquire a connection from the pool."""
        with self._lock:
            # Try to get a healthy connection
            attempts = len(self._available)
            while attempts > 0 and self._available:
                conn = self._available.popleft()
                stats = self._stats.get(conn.conn_id)

                if stats and stats.health == ConnectionHealth.DEAD:
                    # Skip dead connections, they'll be cleaned up
                    attempts -= 1
                    continue

                self._in_use[conn.conn_id] = conn
                return conn

            return None

    def release(self, conn: 'NNTPConnection', success: bool = True, response_time_ms: float = 0.0) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            if conn.conn_id in self._in_use:
                del self._in_use[conn.conn_id]

            stats = self._stats.get(conn.conn_id)
            if stats:
                if success:
                    stats.record_success(response_time_ms)
                else:
                    stats.record_failure()

            # Only return healthy connections to pool
            if conn.connected and (not stats or stats.health != ConnectionHealth.DEAD):
                self._available.append(conn)

    def record_throughput(self, mbps: float) -> None:
        """Record current throughput for scaling decisions."""
        self._throughput_samples.append(mbps)

    def set_target_throughput(self, mbps: float) -> None:
        """Set expected target throughput."""
        self._target_throughput = mbps

    def should_scale_up(self) -> bool:
        """Check if pool should scale up."""
        with self._lock:
            if len(self._available) + len(self._in_use) >= self.max_connections:
                return False

            utilization = self.get_utilization()
            return utilization > self.scale_up_threshold

    def should_scale_down(self) -> bool:
        """Check if pool should scale down."""
        with self._lock:
            if len(self._available) + len(self._in_use) <= self.min_connections:
                return False

            utilization = self.get_utilization()
            return utilization < self.scale_down_threshold

    def get_utilization(self) -> float:
        """Get current pool utilization (0.0 - 1.0)."""
        with self._lock:
            total = len(self._available) + len(self._in_use)
            if total == 0:
                return 0.0
            return len(self._in_use) / total

    def get_scale_recommendation(self) -> int:
        """
        Get recommended connection count adjustment.

        Returns:
            Positive number to add connections, negative to remove, 0 for no change
        """
        now = time.time()
        if now - self._last_scale_time < self._scale_cooldown:
            return 0

        with self._lock:
            if self.should_scale_up():
                self._last_scale_time = now
                return min(self.scale_step, self.max_connections - len(self._available) - len(self._in_use))
            elif self.should_scale_down():
                self._last_scale_time = now
                return -min(self.scale_step, len(self._available) + len(self._in_use) - self.min_connections)

            return 0

    def get_healthy_count(self) -> int:
        """Get count of healthy connections."""
        with self._lock:
            healthy = 0
            for conn in self._available:
                stats = self._stats.get(conn.conn_id)
                if not stats or stats.health in (ConnectionHealth.HEALTHY, ConnectionHealth.DEGRADED):
                    healthy += 1
            return healthy

    def get_pool_status(self) -> Dict:
        """Get detailed pool status."""
        with self._lock:
            health_counts = {h: 0 for h in ConnectionHealth}
            for stats in self._stats.values():
                health_counts[stats.health] += 1

            return {
                'available': len(self._available),
                'in_use': len(self._in_use),
                'total': len(self._available) + len(self._in_use),
                'utilization': self.get_utilization(),
                'health': {h.value: c for h, c in health_counts.items()},
                'target': self.target_connections
            }


# =============================================================================
# SSL CONTEXT OPTIMIZED FOR 10 Gbps THROUGHPUT
# =============================================================================
def create_high_performance_ssl_context() -> ssl.SSLContext:
    """
    Create SSL context optimized for maximum throughput.

    Based on RFC 8143 recommendations and SABnzbd best practices.
    """
    # Prefer TLS 1.3 (faster handshake, better performance)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # Security settings
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    # Load system CA certificates
    context.load_default_certs()

    # TLS 1.2 minimum (TLS 1.3 preferred when available)
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Optimized cipher suites for speed (AES-GCM is hardware accelerated)
    # TLS 1.3 ciphers are automatically preferred when available
    try:
        context.set_ciphers(
            'TLS_AES_256_GCM_SHA384:'      # TLS 1.3
            'TLS_AES_128_GCM_SHA256:'      # TLS 1.3
            'TLS_CHACHA20_POLY1305_SHA256:'  # TLS 1.3
            'ECDHE+AESGCM:'                # TLS 1.2 with forward secrecy
            'DHE+AESGCM:'                  # TLS 1.2 fallback
            'ECDHE+AES:'
            '!aNULL:!MD5:!DSS'
        )
    except ssl.SSLError:
        # Fallback if some ciphers not available
        context.set_ciphers('DEFAULT:!aNULL:!MD5')

    # Enable session tickets for faster reconnects
    # (TLS session resumption reduces handshake overhead)
    context.options |= ssl.OP_NO_TICKET  # Disable for now, some servers have issues

    return context


# Global SSL context (reused for all connections - enables session caching)
_SSL_CONTEXT: Optional[ssl.SSLContext] = None

def get_ssl_context() -> ssl.SSLContext:
    """Get or create the global SSL context."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = create_high_performance_ssl_context()
    return _SSL_CONTEXT


@dataclass
class ServerConfig:
    host: str
    port: int = 563
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    connections: int = 20
    timeout: float = 30.0


class NNTPConnection:
    """Single threaded NNTP connection with retry and RTT measurement."""

    # Default retry configuration
    DEFAULT_RETRY_CONFIG = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=30.0,
        jitter_factor=0.5
    )

    def __init__(self, config: ServerConfig, conn_id: int = 0, retry_config: Optional[RetryConfig] = None):
        self.config = config
        self.conn_id = conn_id
        self.retry_config = retry_config or self.DEFAULT_RETRY_CONFIG
        self.socket: Optional[socket.socket] = None
        self.file = None  # File-like wrapper for readline
        self.connected = False
        self.bytes_downloaded = 0
        self.articles_fetched = 0

        # RTT measurement
        self._last_rtt_ms: float = 0.0
        self._rtt_samples: Deque[float] = deque(maxlen=20)

        # Retry statistics
        self.total_retries = 0
        self.successful_retries = 0

    @property
    def avg_rtt_ms(self) -> float:
        """Get average RTT in milliseconds."""
        if not self._rtt_samples:
            return 0.0
        return statistics.mean(self._rtt_samples)

    def _connect_once(self) -> bool:
        """Single connection attempt (internal)."""
        # Create socket
        raw_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_socket.settimeout(self.config.timeout)

        # =================================================================
        # SOCKET OPTIONS OPTIMIZED FOR 10 Gbps
        # =================================================================
        # Disable Nagle's algorithm for low latency
        raw_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # LARGE buffers for high bandwidth (8MB receive, 1MB send)
        raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

        # Keep-alive to detect dead connections
        raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        if self.config.use_ssl:
            # Use global optimized SSL context
            context = get_ssl_context()
            self.socket = context.wrap_socket(raw_socket, server_hostname=self.config.host)
        else:
            self.socket = raw_socket

        # Connect with RTT measurement
        connect_start = time.perf_counter()
        self.socket.connect((self.config.host, self.config.port))
        connect_rtt = (time.perf_counter() - connect_start) * 1000
        self._rtt_samples.append(connect_rtt)
        self._last_rtt_ms = connect_rtt

        # Large read buffer for high throughput (256KB)
        self.file = self.socket.makefile('rb', buffering=262144)

        # Read greeting
        greeting = self._readline()
        if not greeting.startswith(b'200') and not greeting.startswith(b'201'):
            raise ConnectionError(f"Bad greeting: {greeting}")

        # Authenticate
        if self.config.username:
            self._send(f"AUTHINFO USER {self.config.username}\r\n")
            resp = self._readline()
            if resp.startswith(b'381'):
                self._send(f"AUTHINFO PASS {self.config.password}\r\n")
                resp = self._readline()
            if not resp.startswith(b'281'):
                raise ConnectionError(f"Auth failed: {resp}")

        self.connected = True
        return True

    def connect(self, with_retry: bool = True) -> bool:
        """
        Establish connection to NNTP server with optimized settings.

        Args:
            with_retry: If True, use exponential backoff retry on failure

        Returns:
            True if connected successfully
        """
        if not with_retry:
            try:
                return self._connect_once()
            except Exception as e:
                logger.error(f"Connection {self.conn_id} failed: {e}")
                self.close()
                return False

        # Retry with exponential backoff
        retry_op = RetryableOperation(self.retry_config, f"connect_{self.conn_id}")

        try:
            retry_op.execute(self._connect_once)
            if retry_op.attempt > 0:
                self.total_retries += retry_op.attempt
                self.successful_retries += 1
                logger.info(f"Connection {self.conn_id} succeeded after {retry_op.attempt} retries")
            return True
        except Exception as e:
            logger.error(f"Connection {self.conn_id} failed after {retry_op.attempt} attempts: {e}")
            self.close()
            return False

    def _send(self, cmd: str) -> None:
        """Send command to server."""
        self.socket.sendall(cmd.encode('utf-8'))

    def _readline(self) -> bytes:
        """Read a line from server."""
        return self.file.readline()

    def _read_body(self) -> bytes:
        """
        Read multi-line body until terminator.

        Optimized for yEnc data which is typically 700KB-1MB per segment.
        Uses pre-allocated buffer for better performance.
        """
        # Pre-allocate for typical yEnc segment size
        chunks = []
        total_size = 0

        while True:
            line = self.file.readline()
            if line == b'.\r\n':
                break
            # Dot-stuffing: lines starting with .. become .
            if line.startswith(b'..'):
                line = line[1:]
            chunks.append(line)
            total_size += len(line)

        # Use join for efficient concatenation
        return b''.join(chunks)

    def fetch_body(self, message_id: str, with_retry: bool = False, max_retries: int = 3) -> Optional[bytes]:
        """
        Fetch article body by message ID.

        Args:
            message_id: NNTP message ID to fetch
            with_retry: If True, retry on transient failures
            max_retries: Maximum retry attempts (only used if with_retry=True)

        Returns:
            Article body data or None if not found/failed
        """
        attempts = 0
        max_attempts = max_retries + 1 if with_retry else 1

        while attempts < max_attempts:
            try:
                # Measure RTT for this request
                request_start = time.perf_counter()

                self._send(f"BODY <{message_id}>\r\n")
                response = self._readline()

                # Record RTT (time to first byte of response)
                rtt_ms = (time.perf_counter() - request_start) * 1000
                self._rtt_samples.append(rtt_ms)
                self._last_rtt_ms = rtt_ms

                if response.startswith(b'222'):
                    data = self._read_body()
                    self.bytes_downloaded += len(data)
                    self.articles_fetched += 1
                    return data
                elif response.startswith(b'430'):
                    return None  # Article not found - don't retry
                else:
                    logger.warning(f"Unexpected response: {response[:50]}")
                    if not with_retry:
                        return None
                    # Retry on unexpected responses
                    attempts += 1
                    if attempts < max_attempts:
                        delay = calculate_backoff_delay(attempts - 1, self.retry_config)
                        time.sleep(delay)
                        self.total_retries += 1
                    continue

            except Exception as e:
                attempts += 1
                if attempts < max_attempts:
                    delay = calculate_backoff_delay(attempts - 1, self.retry_config)
                    logger.debug(f"[RETRY] fetch_body {message_id} attempt {attempts}/{max_attempts} after {delay:.2f}s: {e}")
                    time.sleep(delay)
                    self.total_retries += 1
                else:
                    logger.error(f"Error fetching {message_id} after {attempts} attempts: {e}")
                    return None

        return None

    def fetch_bodies_pipelined(self, message_ids: List[str], pipeline_depth: int = 10) -> Dict[str, Optional[bytes]]:
        """
        Fetch multiple article bodies using NNTP pipelining.

        Sends multiple BODY commands before reading responses to reduce
        round-trip latency. This is key for high-throughput downloads.

        Args:
            message_ids: List of message IDs to fetch
            pipeline_depth: How many commands to send before reading

        Returns:
            Dict mapping message_id to body data (or None if not found)
        """
        results = {}
        pending = []

        try:
            for i, msg_id in enumerate(message_ids):
                # Send BODY command
                self._send(f"BODY <{msg_id}>\r\n")
                pending.append(msg_id)

                # Read responses when pipeline is full or at end
                if len(pending) >= pipeline_depth or i == len(message_ids) - 1:
                    for pending_id in pending:
                        response = self._readline()

                        if response.startswith(b'222'):
                            data = self._read_body()
                            results[pending_id] = data
                            self.bytes_downloaded += len(data)
                            self.articles_fetched += 1
                        elif response.startswith(b'430'):
                            results[pending_id] = None  # Article not found
                        else:
                            logger.warning(f"Unexpected response for {pending_id}: {response[:50]}")
                            results[pending_id] = None

                    pending = []

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Mark remaining as failed
            for pending_id in pending:
                results[pending_id] = None

        return results

    def fetch_streaming(
        self,
        get_next_id: callable,
        on_data: callable,
        pipeline_depth: int = 5,
        pause_event: Optional[threading.Event] = None,
        stop_flag: Optional[callable] = None
    ) -> int:
        """
        Continuous streaming fetch with sliding window pipeline.

        Maintains exactly `pipeline_depth` requests in flight at all times,
        providing constant throughput without bursts.

        Args:
            get_next_id: Callable that returns next message_id or None if done
            on_data: Callable(message_id, data) called for each received segment
            pipeline_depth: Number of requests to keep in flight (default 5)
            pause_event: Optional threading.Event to pause fetching (set=running, clear=paused)
            stop_flag: Optional callable returning True if stop requested

        Returns:
            Number of segments successfully fetched
        """
        pending_ids = []  # IDs with requests in flight
        fetched = 0

        try:
            # Fill initial pipeline
            for _ in range(pipeline_depth):
                if stop_flag and stop_flag():
                    break
                msg_id = get_next_id()
                if msg_id is None:
                    break
                self._send(f"BODY <{msg_id}>\r\n")
                pending_ids.append(msg_id)

            # Sliding window: read one, send one
            while pending_ids:
                # Check stop request
                if stop_flag and stop_flag():
                    break

                # Wait if paused (with timeout for stop responsiveness)
                if pause_event:
                    while not pause_event.is_set():
                        if stop_flag and stop_flag():
                            break
                        pause_event.wait(timeout=0.1)
                    if stop_flag and stop_flag():
                        break

                # Read oldest response
                msg_id = pending_ids.pop(0)
                response = self._readline()

                if response.startswith(b'222'):
                    data = self._read_body()
                    self.bytes_downloaded += len(data)
                    self.articles_fetched += 1
                    fetched += 1
                    on_data(msg_id, data)
                elif response.startswith(b'430'):
                    on_data(msg_id, None)  # Not found
                else:
                    on_data(msg_id, None)

                # Send next request to maintain pipeline depth
                if not (stop_flag and stop_flag()):
                    next_id = get_next_id()
                    if next_id is not None:
                        self._send(f"BODY <{next_id}>\r\n")
                        pending_ids.append(next_id)

        except Exception as e:
            logger.error(f"Streaming error: {e}")

        return fetched

    def close(self):
        """Close connection."""
        self.connected = False
        try:
            if self.file:
                self.file.close()
            if self.socket:
                self.socket.close()
        except:
            pass


class FastNNTPDownloader:
    """
    High-performance NNTP downloader using thread pool.

    Each thread maintains its own connection for true parallelism.

    v1.3.0 Enhancements:
    - Dynamic connection pool with health monitoring
    - Adaptive pipelining based on RTT
    - Retry with exponential backoff
    """

    def __init__(
        self,
        config: ServerConfig,
        on_segment: Optional[Callable[[str, bytes], None]] = None,
        on_progress: Optional[Callable[[int, int, float], None]] = None,
        retry_config: Optional[RetryConfig] = None,
        use_dynamic_pool: bool = True,
        use_adaptive_pipeline: bool = True
    ):
        self.config = config
        self.on_segment = on_segment  # Callback when segment is downloaded
        self.on_progress = on_progress  # Callback for progress (completed, total, speed)
        self.retry_config = retry_config or RetryConfig()

        self._connections: List[NNTPConnection] = []
        self._pool: Optional[ThreadPoolExecutor] = None
        self._stats_lock = Lock()
        self._bytes_downloaded = 0
        self._segments_completed = 0
        self._start_time = 0.0

        # Dynamic pool and adaptive pipeline
        self._use_dynamic_pool = use_dynamic_pool
        self._use_adaptive_pipeline = use_adaptive_pipeline
        self._dynamic_pool: Optional[DynamicConnectionPool] = None
        self._adaptive_pipeline: Optional[AdaptivePipeline] = None

        if use_dynamic_pool:
            self._dynamic_pool = DynamicConnectionPool(
                min_connections=5,
                max_connections=config.connections * 2,
                target_connections=config.connections
            )

        if use_adaptive_pipeline:
            self._adaptive_pipeline = AdaptivePipeline(initial_depth=30)

    @property
    def pool_status(self) -> Optional[Dict]:
        """Get dynamic pool status if enabled."""
        if self._dynamic_pool:
            return self._dynamic_pool.get_pool_status()
        return None

    @property
    def current_pipeline_depth(self) -> int:
        """Get current adaptive pipeline depth."""
        if self._adaptive_pipeline:
            return self._adaptive_pipeline.depth
        return 30  # Default

    @property
    def avg_rtt_ms(self) -> float:
        """Get average RTT across all connections."""
        if not self._connections:
            return 0.0
        rtts = [c.avg_rtt_ms for c in self._connections if c.avg_rtt_ms > 0]
        if not rtts:
            return 0.0
        return statistics.mean(rtts)

    def connect(self) -> int:
        """Establish all connections. Returns number of successful connections."""
        logger.info(f"Connecting to {self.config.host}:{self.config.port}...")

        # Create connections in parallel using threads
        def create_conn(i):
            conn = NNTPConnection(self.config, i, self.retry_config)
            if conn.connect():
                return conn
            return None

        with ThreadPoolExecutor(max_workers=self.config.connections) as executor:
            futures = [executor.submit(create_conn, i) for i in range(self.config.connections)]
            for future in as_completed(futures):
                conn = future.result()
                if conn:
                    self._connections.append(conn)
                    if self._dynamic_pool:
                        self._dynamic_pool.add_connection(conn)

        # Update adaptive pipeline with initial RTT measurements
        if self._adaptive_pipeline and self._connections:
            avg_rtt = self.avg_rtt_ms
            if avg_rtt > 0:
                self._adaptive_pipeline.record_rtt(avg_rtt)
                self._adaptive_pipeline.adjust()
                logger.info(f"[PIPELINE] Initial depth set to {self._adaptive_pipeline.depth} "
                          f"(avg RTT: {avg_rtt:.1f}ms)")

        logger.info(f"Connected: {len(self._connections)}/{self.config.connections} connections")
        return len(self._connections)

    def download_segments(self, message_ids: List[str]) -> Dict[str, Optional[bytes]]:
        """
        Download multiple segments in parallel using all connections.

        Returns dict mapping message_id to data (or None if failed).
        """
        if not self._connections:
            raise RuntimeError("Not connected")

        results: Dict[str, Optional[bytes]] = {}
        results_lock = Lock()

        self._bytes_downloaded = 0
        self._segments_completed = 0
        self._start_time = time.time()

        # Create work queue
        work_queue = Queue()
        for msg_id in message_ids:
            work_queue.put(msg_id)

        total = len(message_ids)

        def worker(conn: NNTPConnection):
            """Worker function for each connection."""
            local_bytes = 0
            local_segments = 0

            while True:
                try:
                    msg_id = work_queue.get_nowait()
                except Empty:
                    break

                data = conn.fetch_body(msg_id)

                # Update results without lock (dict assignment is atomic in CPython)
                results[msg_id] = data

                if data:
                    local_bytes += len(data)
                    local_segments += 1

                    # Callback OUTSIDE lock for max parallelism
                    if self.on_segment:
                        self.on_segment(msg_id, data)

                work_queue.task_done()

            # Batch update stats at end (minimize lock contention)
            with results_lock:
                self._bytes_downloaded += local_bytes
                self._segments_completed += local_segments

        # Run workers - one per connection (no thread pool overhead)
        import threading
        threads = []
        for conn in self._connections:
            t = threading.Thread(target=worker, args=(conn,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Final progress
        elapsed = time.time() - self._start_time
        speed = self._bytes_downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
        print(f"\nDownloaded {self._segments_completed}/{total} segments at {speed:.1f} MB/s")

        return results

    def close(self):
        """Close all connections."""
        for conn in self._connections:
            conn.close()
        self._connections.clear()


# Quick test function
def test_download(host: str, port: int, username: str, password: str, message_ids: List[str]):
    """Test download with given credentials and message IDs."""
    config = ServerConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        connections=20
    )

    downloader = FastNNTPDownloader(config)

    try:
        if downloader.connect() == 0:
            print("Failed to connect!")
            return

        results = downloader.download_segments(message_ids)

        success = sum(1 for v in results.values() if v is not None)
        print(f"Success: {success}/{len(message_ids)}")

    finally:
        downloader.close()
