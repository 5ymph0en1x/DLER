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
- Adaptive pipelining based on throughput (AIMD)
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


# Optional native, GIL-released dot-unstuffing for the per-body copy. Falls back
# to pure Python if the compiled module isn't importable.
_native_unstuff = None
try:
    import yenc_turbo as _yenc_native  # type: ignore
    _native_unstuff = getattr(_yenc_native, "unstuff_body", None)
except Exception:
    try:
        import os as _os, sys as _sys
        _nd = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "native")
        if _nd not in _sys.path:
            _sys.path.insert(0, _nd)
        import yenc_turbo as _yenc_native  # type: ignore
        _native_unstuff = getattr(_yenc_native, "unstuff_body", None)
    except Exception:
        _native_unstuff = None

# =============================================================================
# NNTP RESPONSE CLASSIFICATION
# =============================================================================
RESP_OK = "ok"            # 222: body follows
RESP_MISSING = "missing"  # 430: no such article (real gap -> PAR2)
RESP_FATAL = "fatal"      # anything else: desync/dying connection


def classify_response(line: bytes) -> str:
    """Classify the first line of a BODY response.

    Only 222 (ok) and 430 (missing) leave the connection usable. Everything
    else -- 4xx/5xx service/auth/permission errors, an empty read (TCP drop),
    or non-numeric garbage -- means the connection is desynced or dying.
    """
    if not line:
        return RESP_FATAL
    if line.startswith(b"222"):
        return RESP_OK
    if line.startswith(b"430"):
        return RESP_MISSING
    return RESP_FATAL


class PipelineError(Exception):
    """Raised by fetch_streaming after a fatal pipeline condition.

    By the time this is raised, in-flight ids have been requeued and the
    connection has been closed.
    """


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
# ADAPTIVE PIPELINING BASED ON THROUGHPUT AIMD
# =============================================================================

class AdaptivePipeline:
    """Adapts pipeline depth via throughput AIMD.

    Additive-increase while throughput keeps rising; multiplicative-decrease on
    a throughput regression or an error spike; hold on a plateau. RTT is still
    recorded (for the on-screen metric) but no longer drives depth -- in a
    saturated pipeline the send->response time is sojourn time (~ depth x service
    time), not RTT, so using it to set depth causes runaway growth.
    """

    MIN_DEPTH = 5
    MAX_DEPTH = 100
    DEFAULT_DEPTH = 30

    def __init__(self, initial_depth: int = DEFAULT_DEPTH, min_depth: int = MIN_DEPTH,
                 max_depth: int = MAX_DEPTH, increase_step: int = 8,
                 decrease_factor: float = 0.7, epsilon_mbps: float = 10.0,
                 adjustment_interval: float = 1.0, sample_size: int = 8):
        self._min_depth = max(1, min_depth)
        self._max_depth = max(self._min_depth, max_depth)
        self._depth = max(self._min_depth, min(self._max_depth, initial_depth))
        self._increase_step = increase_step
        self._decrease_factor = decrease_factor
        self._epsilon = epsilon_mbps
        self._adjustment_interval = adjustment_interval
        self._error_threshold = 0.05  # >5% errors => back off

        self._tput_samples: Deque[float] = deque(maxlen=sample_size)
        self._rtt_samples: Deque[float] = deque(maxlen=20)
        self._baseline: Optional[float] = None
        self._last_adjustment_time = 0.0
        self._lock = Lock()

    @property
    def depth(self) -> int:
        return self._depth

    def record_throughput(self, mbps: float) -> None:
        with self._lock:
            self._tput_samples.append(mbps)

    def record_rtt(self, rtt_ms: float) -> None:
        with self._lock:
            self._rtt_samples.append(rtt_ms)

    def get_avg_rtt(self) -> Optional[float]:
        with self._lock:
            if not self._rtt_samples:
                return None
            return statistics.mean(self._rtt_samples)

    def reset_baseline(self) -> None:
        """Forget the comparison point (after a pool scale or pause/resume)."""
        with self._lock:
            self._baseline = None

    def adjust(self, error_rate: float = 0.0) -> int:
        now = time.time()
        with self._lock:
            if now - self._last_adjustment_time < self._adjustment_interval:
                return self._depth
            if not self._tput_samples:
                return self._depth
            self._last_adjustment_time = now
            avg = statistics.mean(self._tput_samples)

            if self._baseline is None:
                self._baseline = avg
                return self._depth

            old = self._depth
            if error_rate > self._error_threshold:
                self._depth = max(self._min_depth, int(self._depth * self._decrease_factor))
            elif avg > self._baseline + self._epsilon:
                self._depth = min(self._max_depth, self._depth + self._increase_step)
            elif avg < self._baseline - self._epsilon:
                self._depth = max(self._min_depth, int(self._depth * self._decrease_factor))
            # else: plateau -> hold

            self._baseline = avg
            if self._depth != old:
                logger.debug(f"[PIPELINE] depth {old} -> {self._depth} "
                             f"(tput {avg:.1f} MB/s, err {error_rate:.2f})")
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
    """Tracks connection health and recommends scaling for the streaming engine.

    Scaling is driven by work-queue backlog vs. active connection count (not the
    borrow/return utilization of the old design), bounded by [min, max] where
    max is the provider connection cap. Per-connection health comes from
    ConnectionStats fed via record_result().
    """

    def __init__(self, min_connections: int = 5, max_connections: int = 20,
                 target_connections: int = 20, scale_step: int = 2,
                 scale_cooldown: float = 10.0, up_backlog_factor: float = 3.0,
                 down_backlog_factor: float = 1.0):
        self.min_connections = max(1, min_connections)
        self.max_connections = max(self.min_connections, max_connections)
        self.target_connections = max(self.min_connections,
                                      min(self.max_connections, target_connections))
        self.scale_step = max(1, scale_step)
        self.scale_cooldown = scale_cooldown
        self.up_backlog_factor = up_backlog_factor
        self.down_backlog_factor = down_backlog_factor

        self._stats: Dict[int, ConnectionStats] = {}
        self._lock = RLock()
        self._last_scale_time = 0.0

    def add_connection(self, conn: 'NNTPConnection') -> None:
        with self._lock:
            self._stats[conn.conn_id] = ConnectionStats(conn_id=conn.conn_id)

    def remove_connection(self, conn_id: int) -> None:
        with self._lock:
            self._stats.pop(conn_id, None)

    def record_result(self, conn_id: int, success: bool,
                      response_time_ms: float = 0.0) -> None:
        with self._lock:
            stats = self._stats.get(conn_id)
            if stats is None:
                stats = self._stats[conn_id] = ConnectionStats(conn_id=conn_id)
            if success:
                stats.record_success(response_time_ms)
            else:
                stats.record_failure()

    def total(self) -> int:
        with self._lock:
            return len(self._stats)

    def healthy_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._stats.values()
                       if s.health in (ConnectionHealth.HEALTHY, ConnectionHealth.DEGRADED))

    def dead_conn_ids(self) -> List[int]:
        with self._lock:
            return [cid for cid, s in self._stats.items()
                    if s.health == ConnectionHealth.DEAD]

    def should_scale_up(self, backlog: int, active: int) -> bool:
        if active >= self.max_connections:
            return False
        return backlog > active * self.up_backlog_factor

    def should_scale_down(self, backlog: int, active: int) -> bool:
        if active <= self.min_connections:
            return False
        return backlog < active * self.down_backlog_factor

    def get_scale_recommendation(self, backlog: int, active: int,
                                 now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            if now - self._last_scale_time < self.scale_cooldown:
                return 0
            if self.should_scale_up(backlog, active):
                self._last_scale_time = now
                room = self.max_connections - active
                return min(self.scale_step, room)
            if self.should_scale_down(backlog, active):
                self._last_scale_time = now
                room = active - self.min_connections
                return -min(self.scale_step, room)
            return 0

    def get_pool_status(self) -> Dict:
        with self._lock:
            health_counts = {h: 0 for h in ConnectionHealth}
            for s in self._stats.values():
                health_counts[s.health] += 1
            return {
                'total': len(self._stats),
                'healthy': self.healthy_count(),
                'min': self.min_connections,
                'max': self.max_connections,
                'health': {h.value: c for h, c in health_counts.items()},
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

    # Enable session tickets for faster reconnects.
    # (TLS session resumption reduces handshake overhead; the global cached
    #  context below reuses sessions across connections.)
    # Session resumption is left ON; if a specific server misbehaves, set
    #  context.options |= ssl.OP_NO_TICKET here.

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
        # Reusable recv buffer (recv_into, no per-recv alloc). Cursor model:
        # unconsumed data is buf[_rstart:_rend]; reset/compacted as needed.
        self._rbuf = bytearray(1 << 20)
        self._rstart = 0
        self._rend = 0
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

    def _apply_socket_options(self, sock: socket.socket) -> None:
        """Apply high-performance socket options to a connected socket."""
        sock.settimeout(self.config.timeout)

        # =================================================================
        # SOCKET OPTIONS OPTIMIZED FOR 10 Gbps
        # =================================================================
        # Disable Nagle's algorithm for low latency
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # LARGE buffers for high bandwidth (8MB receive, 1MB send)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

        # Keep-alive to detect dead connections
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def _open_raw_socket(self) -> socket.socket:
        """Resolve host with AF_UNSPEC and connect, trying each address in order.

        Tries IPv6 first if present, falls back to IPv4 on OSError.  Returns
        the first successfully-connected raw socket (no SSL wrapping yet).
        """
        addrinfos = socket.getaddrinfo(
            self.config.host, self.config.port,
            socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        last_exc: Optional[Exception] = None
        for family, socktype, _proto, _canonname, sockaddr in addrinfos:
            sock = socket.socket(family, socktype, _proto)
            try:
                self._apply_socket_options(sock)  # timeout + options BEFORE connect
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                sock.close()
                last_exc = exc
        raise OSError(
            f"Could not connect to {self.config.host}:{self.config.port}"
        ) from last_exc

    def _connect_once(self) -> bool:
        """Single connection attempt (internal)."""
        # Resolve and connect across IPv6/IPv4, measuring connect RTT
        connect_start = time.perf_counter()
        raw_socket = self._open_raw_socket()
        connect_rtt = (time.perf_counter() - connect_start) * 1000
        self._rtt_samples.append(connect_rtt)
        self._last_rtt_ms = connect_rtt

        if self.config.use_ssl:
            # Use global optimized SSL context
            context = get_ssl_context()
            self.socket = context.wrap_socket(raw_socket, server_hostname=self.config.host)
        else:
            self.socket = raw_socket

        # Reset read buffer for this connection (block recv, no makefile)
        self._rstart = 0
        self._rend = 0

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

    def _fill(self) -> bool:
        """Read more bytes into the reusable buffer via recv_into (no per-recv
        allocation, no extend copy). Returns False on EOF.

        Buffer layout: unconsumed bytes live in self._rbuf[_rstart:_rend]. When
        fully consumed we reset to the front; when the tail is full we compact
        (and grow x2 only if a single body exceeds the buffer).
        """
        buf = self._rbuf
        if self._rstart == self._rend:
            self._rstart = 0
            self._rend = 0
        elif self._rend == len(buf):
            avail = self._rend - self._rstart
            buf[:avail] = buf[self._rstart:self._rend]   # compact to front
            self._rstart = 0
            self._rend = avail
            if self._rend == len(buf):
                buf.extend(bytearray(len(buf)))           # grow x2 (rare)
        n = self.socket.recv_into(memoryview(buf)[self._rend:])
        if not n:
            return False
        self._rend += n
        return True

    def _readline(self) -> bytes:
        """Read one line (including the trailing \\n) from the buffered stream.

        Returns b'' on EOF before a newline (the caller treats that as fatal).
        Shares the buffer with _read_body so block body reads never desync
        against status-line reads on the same socket.
        """
        while True:
            idx = self._rbuf.find(b'\n', self._rstart, self._rend)
            if idx != -1:
                line = bytes(self._rbuf[self._rstart:idx + 1])
                self._rstart = idx + 1
                return line
            if not self._fill():
                line = bytes(self._rbuf[self._rstart:self._rend])
                self._rstart = self._rend
                return line

    def _extract_body(self, start: int, end: int) -> bytes:
        """Copy self._rbuf[start:end] with NNTP dot-unstuffing. Uses the native
        GIL-released path when available, else a pure-Python fallback."""
        if _native_unstuff is not None:
            return _native_unstuff(self._rbuf, start, end)
        data = bytes(self._rbuf[start:end])
        data = data.replace(b'\r\n..', b'\r\n.')
        if data[:2] == b'..':
            data = data[1:]
        return data

    def _read_body(self) -> bytes:
        """Read a dot-terminated multi-line body in BLOCKS (not per line).

        recv_into the reusable buffer, INCREMENTALLY scan for the terminating
        '.' line (each fill rescans only the new bytes + a 4-byte straddle
        margin -> O(total bytes), not O(N^2)), then hand the body off to the
        native GIL-released unstuff+copy. `scanned` is tracked RELATIVE to
        _rstart so it survives buffer compaction inside _fill().
        """
        # Need >= 3 bytes to test for an empty body ('.' line at the start).
        while self._rend - self._rstart < 3:
            if not self._fill():
                raise ConnectionError("EOF while reading article body")
        if self._rbuf[self._rstart:self._rstart + 3] == b'.\r\n':
            self._rstart += 3
            return b''

        scanned = 0  # bytes from _rstart already confirmed terminator-free (minus margin)
        while True:
            start = self._rstart
            idx = self._rbuf.find(b'\r\n.\r\n', start + scanned, self._rend)
            if idx != -1:
                body = self._extract_body(start, idx + 2)   # keep last data line's \r\n
                self._rstart = idx + 5                       # consume through '.' terminator
                return body
            grown = (self._rend - start) - 4
            scanned = grown if grown > 0 else 0
            if not self._fill():
                raise ConnectionError("EOF while reading article body")

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

    def fetch_streaming(
        self,
        get_next_id,
        on_data,
        depth_provider,
        requeue,
        pause_event: Optional[threading.Event] = None,
        stop_flag: Optional[Callable] = None,
        on_result: Optional[Callable] = None,
        retire_flag: Optional[Callable] = None,
    ) -> int:
        """Continuous streaming fetch with a live, adjustable sliding window.

        The in-flight window is topped up toward ``depth_provider()`` after each
        response, so adaptive depth changes are applied in flight. On any fatal
        response (see classify_response) or exception, every unread in-flight id
        is handed to ``requeue`` and the connection is closed before raising
        PipelineError, so the caller can retry those ids on another connection.

        Args:
            get_next_id: returns next message_id or None when the queue is empty.
            on_data: on_data(message_id, data | None) for delivered/missing segments.
            depth_provider: returns the current target in-flight depth (read live).
            requeue: requeue(list_of_message_ids) for ids that were sent but not
                resolved (so they are retried, not lost).
            pause_event: set=running / clear=paused.
            stop_flag: returns True if the whole download is stopping.
            on_result: optional on_result(success: bool, response_time_ms: float)
                for per-connection health tracking.
            retire_flag: returns True if this worker should stop sending new
                requests, drain its window, and return (graceful scale-down).

        Returns:
            Number of segments successfully fetched (222).
        """
        pending: Deque[str] = deque()
        fetched = 0
        fill_start: Optional[float] = None  # time the first request of a fill was sent

        def _stopping() -> bool:
            return bool(stop_flag and stop_flag())

        def _retiring() -> bool:
            return bool(retire_flag and retire_flag())

        def _send_body(mid: str) -> None:
            nonlocal fill_start
            if not pending:
                fill_start = time.perf_counter()
            self._send(f"BODY <{mid}>\r\n")
            pending.append(mid)

        def _fill_to_depth() -> None:
            target = max(1, int(depth_provider()))
            while len(pending) < target:
                if _stopping():
                    return
                if pending and _retiring():
                    return
                nxt = get_next_id()
                if nxt is None:
                    return
                _send_body(nxt)

        try:
            _fill_to_depth()

            while pending:
                if _stopping():
                    requeue(list(pending))
                    pending = deque()
                    break

                if pause_event is not None:
                    while not pause_event.is_set():
                        if _stopping():
                            break
                        pause_event.wait(timeout=0.1)
                    if _stopping():
                        requeue(list(pending))
                        pending = deque()
                        break

                msg_id = pending.popleft()
                try:
                    response = self._readline()
                except Exception as e:
                    requeue([msg_id, *pending])
                    pending = deque()
                    self.close()
                    if on_result:
                        on_result(False, 0.0)
                    raise PipelineError(f"readline failed: {e}")

                if fill_start is not None:
                    rtt = (time.perf_counter() - fill_start) * 1000
                    self._rtt_samples.append(rtt)
                    self._last_rtt_ms = rtt
                    fill_start = None

                cls = classify_response(response)
                if cls == RESP_OK:
                    t0 = time.perf_counter()
                    try:
                        data = self._read_body()
                    except Exception as e:
                        requeue([msg_id, *pending])
                        pending = deque()
                        self.close()
                        if on_result:
                            on_result(False, 0.0)
                        raise PipelineError(f"read_body failed: {e}")
                    self.bytes_downloaded += len(data)
                    self.articles_fetched += 1
                    fetched += 1
                    on_data(msg_id, data)
                    if on_result:
                        on_result(True, (time.perf_counter() - t0) * 1000)
                elif cls == RESP_MISSING:
                    on_data(msg_id, None)
                    if on_result:
                        on_result(True, 0.0)  # missing article is not a conn fault
                else:  # RESP_FATAL
                    requeue([msg_id, *pending])
                    pending = deque()
                    self.close()
                    if on_result:
                        on_result(False, 0.0)
                    raise PipelineError(f"fatal response: {response[:40]!r}")

                if not _stopping() and not _retiring():
                    _fill_to_depth()
                # if retiring: stop sending; loop drains remaining `pending`.

            return fetched

        except PipelineError:
            raise
        except Exception as e:
            if pending:
                requeue(list(pending))
                pending = deque()
            try:
                self.close()
            except Exception:
                pass
            raise PipelineError(f"streaming error: {e}")

    def close(self):
        """Close connection."""
        self.connected = False
        self._rstart = 0
        self._rend = 0
        try:
            if self.socket:
                self.socket.close()
        except:
            pass

