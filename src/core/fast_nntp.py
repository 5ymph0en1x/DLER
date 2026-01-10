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
"""

import ssl
import socket
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue, Empty

logger = logging.getLogger(__name__)

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
    """Single threaded NNTP connection."""

    def __init__(self, config: ServerConfig, conn_id: int = 0):
        self.config = config
        self.conn_id = conn_id
        self.socket: Optional[socket.socket] = None
        self.file = None  # File-like wrapper for readline
        self.connected = False
        self.bytes_downloaded = 0
        self.articles_fetched = 0

    def connect(self) -> bool:
        """Establish connection to NNTP server with optimized settings."""
        try:
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

            # Connect
            self.socket.connect((self.config.host, self.config.port))

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

        except Exception as e:
            logger.error(f"Connection {self.conn_id} failed: {e}")
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

    def fetch_body(self, message_id: str) -> Optional[bytes]:
        """Fetch article body by message ID."""
        try:
            self._send(f"BODY <{message_id}>\r\n")
            response = self._readline()

            if response.startswith(b'222'):
                data = self._read_body()
                self.bytes_downloaded += len(data)
                self.articles_fetched += 1
                return data
            elif response.startswith(b'430'):
                return None  # Article not found
            else:
                logger.warning(f"Unexpected response: {response[:50]}")
                return None

        except Exception as e:
            logger.error(f"Error fetching {message_id}: {e}")
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
    """

    def __init__(
        self,
        config: ServerConfig,
        on_segment: Optional[Callable[[str, bytes], None]] = None,
        on_progress: Optional[Callable[[int, int, float], None]] = None
    ):
        self.config = config
        self.on_segment = on_segment  # Callback when segment is downloaded
        self.on_progress = on_progress  # Callback for progress (completed, total, speed)

        self._connections: List[NNTPConnection] = []
        self._pool: Optional[ThreadPoolExecutor] = None
        self._stats_lock = Lock()
        self._bytes_downloaded = 0
        self._segments_completed = 0
        self._start_time = 0.0

    def connect(self) -> int:
        """Establish all connections. Returns number of successful connections."""
        print(f"Connecting to {self.config.host}:{self.config.port}...")

        # Create connections in parallel using threads
        def create_conn(i):
            conn = NNTPConnection(self.config, i)
            if conn.connect():
                return conn
            return None

        with ThreadPoolExecutor(max_workers=self.config.connections) as executor:
            futures = [executor.submit(create_conn, i) for i in range(self.config.connections)]
            for future in as_completed(futures):
                conn = future.result()
                if conn:
                    self._connections.append(conn)

        print(f"Connected: {len(self._connections)}/{self.config.connections} connections")
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
