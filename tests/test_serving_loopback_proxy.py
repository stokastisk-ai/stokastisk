"""The SDK's HTTP client never routes LOOPBACK requests through an HTTP proxy (CPU only; a local fake server, a dead proxy address)."""
import os, pathlib, subprocess, sys, textwrap, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = textwrap.dedent("""
    import sys, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    sys.path.insert(0, sys.argv[1])
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            body = b"ok" if self.path == "/health" else b"vllm:num_requests_running 0\\n"
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    from stokastisk import serving
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    print("metrics", serving.fetch_metrics(base))
""")


class LoopbackIgnoresProxy(unittest.TestCase):
    def run_with(self, extra):
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")}
        env.update(extra, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT / "sdk")], capture_output=True, text=True, env=env, timeout=60)

    def test_fetch_metrics_from_loopback_works_with_a_dead_proxy_configured(self):
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy"):
            with self.subTest(var):
                r = self.run_with({var: "http://127.0.0.1:9"})
                self.assertEqual(r.returncode, 0, r.stderr[-600:]); self.assertIn("vllm:num_requests_running", r.stdout)

    def test_remote_hosts_still_use_the_standard_opener(self):
        sys.path.insert(0, str(ROOT / "sdk"))
        from unittest import mock
        from stokastisk import serving
        with mock.patch.object(serving.urllib.request, "urlopen", return_value="std") as m:
            self.assertEqual(serving._urlopen("https://example.com/x", timeout=1), "std"); m.assert_called_once()


if __name__ == "__main__":
    unittest.main()
