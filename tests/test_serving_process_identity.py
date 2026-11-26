"""ServerProcess process ownership on the published build-5 source (CPU-only, fake server; no vLLM, torch, CUDA or NVML).

sdk/ is byte-identical to the build-5 release source (f02de473). These tests pin what build 5 does. Two KNOWN build-5 limitations are recorded as
expectedFailure: they are mitigated outside the immutable wheel by the orchestration in Enigma (enigma_connector.serverctl) and in the Kaggle
notebooks, never by patching the installed package:
  * stop() called after the server's leader has exited and been reaped can SIGKILL a process group whose number a new session reused;
  * loopback HTTP (health, stream_chat, fetch_metrics) honours http_proxy unless no_proxy lists 127.0.0.1 / localhost."""
import os, pathlib, signal, subprocess, sys, textwrap, time, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from test_serving_runtime import Fake          # noqa: E402  (fake interpreter / fake server)
from stokastisk import serving                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class ProcessIdentity(Fake):
    def sentinel(self):
        s = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"], start_new_session=True)
        self.addCleanup(lambda: (s.kill(), s.wait(5)))
        return s

    def wait_exit(self, s):
        deadline = time.time() + 20
        while s.running and time.time() < deadline:
            time.sleep(0.05)

    def test_server_runs_in_its_own_session_never_the_callers(self):
        s = self.srv(); s.start(); s.wait_ready(20, 0.1)
        self.assertEqual(os.getpgid(s.pid), s.pid); self.assertEqual(os.getsid(s.pid), s.pid)
        self.assertNotIn(s.pid, (os.getpgid(0), os.getsid(0)))

    def test_stop_after_the_server_exited_by_itself_sends_nothing_and_harms_nothing(self):
        guard = self.sentinel(); me = (os.getpid(), os.getpgid(0), os.getsid(0))
        s = self.srv("crash"); s.start(); self.wait_exit(s)
        s.stop(grace=1)
        self.assertEqual(s.shutdown_report["signals"], []); self.assertTrue(s.shutdown_report["group_gone"])
        self.assertTrue(alive(guard.pid)); self.assertEqual(me, (os.getpid(), os.getpgid(0), os.getsid(0)))

    def test_orphaned_children_of_a_dead_leader_are_still_cleaned_up(self):
        s = self.srv("child"); s.start(); s.wait_ready(20, 0.1); time.sleep(0.5)
        pgid = s._pgid
        os.kill(s.pid, signal.SIGKILL); s._proc.wait(5)
        s.stop(grace=1)
        self.assertIn("SIGKILL(group)", s.shutdown_report["signals"]); self.assertTrue(s.shutdown_report["group_gone"])
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_failed_startup_closes_the_log_handle(self):
        s = serving.ServerProcess("m", self.cfg, python="/nonexistent/python", log_path=str(self.td / "x.log"), port=serving.find_free_port())
        with self.assertRaises(Exception):
            s.start()
        self.assertIsNone(s._log); self.assertIsNone(s.stop())

    @unittest.expectedFailure
    def test_KNOWN_BUILD5_LIMITATION_a_reused_pgid_is_signalled_by_stop(self):
        """Build 5: stop() after the leader was reaped probes killpg(pgid, 0) and SIGKILLs whatever group now has that number.
        Mitigation (outside the wheel): never call stop() on a server whose leader already exited; use enigma_connector.serverctl.safe_stop()."""
        s = self.srv("crash"); s.start(); self.wait_exit(s)
        victim = self.sentinel()
        s._pgid = victim.pid                             # simulated PID reuse by an unrelated session leader
        try:
            s.stop(grace=1)
        except Exception:
            pass
        time.sleep(0.3)
        self.assertTrue(alive(victim.pid), "build 5 stop() SIGKILLed an unrelated process group")


class LoopbackProxy(unittest.TestCase):
    SCRIPT = textwrap.dedent("""
        import sys, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        sys.path.insert(0, sys.argv[1])
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b"vllm:num_requests_running 0\\n")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        from stokastisk import serving
        print("metrics", serving.fetch_metrics(f"http://127.0.0.1:{srv.server_address[1]}"))
    """)

    def run_with(self, extra):
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")}
        env.update(extra, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-c", self.SCRIPT, str(ROOT / "sdk")], capture_output=True, text=True, env=env, timeout=60)

    def test_the_documented_mitigation_no_proxy_for_loopback_works_with_build5(self):
        r = self.run_with({"http_proxy": "http://127.0.0.1:9", "no_proxy": "127.0.0.1,localhost,::1"})
        self.assertEqual(r.returncode, 0, r.stderr[-600:]); self.assertIn("vllm:num_requests_running", r.stdout)

    @unittest.expectedFailure
    def test_KNOWN_BUILD5_LIMITATION_loopback_goes_through_a_configured_proxy(self):
        """Build 5: plain urlopen sends 127.0.0.1 traffic to http_proxy when no_proxy does not list it. Mitigation: set no_proxy before starting."""
        r = self.run_with({"http_proxy": "http://127.0.0.1:9"})
        self.assertEqual(r.returncode, 0, r.stderr[-600:])


if __name__ == "__main__":
    unittest.main()
