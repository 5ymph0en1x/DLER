"""Headless throughput tuner. Runs ONE benchmark config (no disk writes) against
the real server and prints a RESULT json line. Orchestrated by tune_sweep.sh.

Usage: uv run python tune.py '<json-spec>'
  spec: {"mode":"single|mp","duration":20,"connections":100,"num_processes":4,
         "turbo":{"pipeline_depth":96,"use_adaptive_pipeline":false,"decoder_threads":0,
                  "writer_threads":8,"max_pipeline_depth":200,"min_connections":100}}
Throughput = median of per-interval download rates over the steady window
(first 45% skipped for ramp/connect, last 10% skipped for teardown).
"""
import sys, os, json, time, threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from core.fast_nntp import ServerConfig
from core.turbo_engine_v2 import TurboEngineV2

NZB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.nzb")
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench_scratch")


def load_server(connections):
    d = json.load(open(os.path.expanduser("~/.dler/config.json")))
    s = d["server"]
    return ServerConfig(host=s["host"], port=s.get("port", 563),
                        username=s.get("username", ""), password=s.get("password", ""),
                        use_ssl=s.get("use_ssl", True), connections=connections,
                        timeout=s.get("timeout", 30))


def steady_mbps(samples):
    """samples: list of (t_seconds, cumulative_bytes). Median of interval rates
    over the middle window."""
    if len(samples) < 5:
        return 0.0
    rates = []
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        db = samples[i][1] - samples[i - 1][1]
        if dt > 0:
            rates.append(db / dt / 1e6)
    n = len(rates)
    lo = int(n * 0.45)            # skip connect + ramp
    hi = max(lo + 1, int(n * 0.90))  # skip teardown
    win = sorted(rates[lo:hi])
    return round(win[len(win) // 2], 1) if win else 0.0


def run_single(server, duration, turbo):
    eng = TurboEngineV2(server, output_dir=SCRATCH, benchmark=True,
                        incremental_verify=False, **turbo)
    samples = []
    th = threading.Thread(target=eng.download_nzb, args=(NZB,), daemon=True)
    t0 = time.time()
    th.start()
    while time.time() - t0 < duration:
        time.sleep(0.5)
        samples.append((time.time() - t0, eng._stats.bytes_downloaded))
    eng.stop()
    th.join(timeout=15)
    eng.disconnect()
    return {"mbps": steady_mbps(samples), "n": len(samples),
            "gb": round(eng._stats.bytes_downloaded / 1e9, 1)}


def run_mp(server, duration, nproc, turbo):
    import multiprocessing as mp
    from core.mp_engine import download_multiprocess
    samples = []
    base = [None]

    def cb(agg):
        if base[0] is None:
            base[0] = time.time()
        samples.append((time.time() - base[0], agg.get("bytes_downloaded", 0)))

    tk = dict(turbo); tk["benchmark"] = True
    stop_ev = mp.get_context("spawn").Event()
    th = threading.Thread(
        target=download_multiprocess,
        kwargs=dict(nzb_path=NZB, server_config=server, output_dir=SCRATCH,
                    num_processes=nproc, turbo_kwargs=tk, progress_cb=cb,
                    poll_interval=0.4, stop_event=stop_ev),
        daemon=True)
    th.start()
    time.sleep(duration)
    stop_ev.set()
    th.join(timeout=25)
    return {"mbps": steady_mbps(samples), "n": len(samples), "nproc": nproc}


def main():
    spec = json.loads(sys.argv[1])
    duration = spec.get("duration", 20)
    connections = spec.get("connections", 100)
    turbo = spec.get("turbo", {})
    server = load_server(connections)
    try:
        if spec.get("mode") == "mp":
            res = run_mp(server, duration, spec.get("num_processes", 4), turbo)
        else:
            res = run_single(server, duration, turbo)
    except Exception as e:
        res = {"error": str(e)}
    res["spec"] = spec
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
