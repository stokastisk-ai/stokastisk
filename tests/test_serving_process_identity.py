"""ServerProcess must only ever signal ITS OWN server's process group (CPU-only, fake server; no vLLM, torch, CUDA or NVML).

Background (Enigma x Kaggle fatal-session investigation, 2026-11-26): every Kaggle process runs as root, so a signal sent to the wrong process group
is never refused by the kernel. After the server's leader has exited and been reaped, its PGID number can be reused by an unrelated new session leader
(any ``start_new_session=True`` child, e.g. the Enigma connector). Linux never hands out a PID that is still in use as a group id, so: once our leader is
reaped, a live process whose PID equals our PGID is NOT ours, while our own orphaned descendants never have that PID."""
import os, pathlib, signal, subprocess, sys, time, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from test_serving_runtime import Fake          # noqa: E402  (reuses the fake interpreter / fake server)


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

    def test_server_runs_in_its_own_session_never_the_callers(self):
        s = self.srv(); s.start(); s.wait_ready(20, 0.1)
        self.assertEqual(os.getpgid(s.pid), s.pid); self.assertEqual(os.getsid(s.pid), s.pid)
        self.assertNotIn(s.pid, (os.getpgid(0), os.getsid(0)))

    def test_stop_after_the_server_exited_by_itself_sends_nothing_and_harms_nothing(self):
        guard = self.sentinel(); me = (os.getpid(), os.getpgid(0), os.getsid(0))
        s = self.srv("crash"); s.start()
        deadline = time.time() + 20
        while s.running and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(s.running)
        s.stop(grace=1)
        self.assertEqual(s.shutdown_report["signals"], []); self.assertTrue(s.shutdown_report["group_gone"])
        self.assertTrue(alive(guard.pid)); self.assertEqual(me, (os.getpid(), os.getpgid(0), os.getsid(0)))

    def test_a_reused_pgid_owned_by_an_unrelated_session_is_never_signalled(self):
        s = self.srv("crash"); s.start()
        deadline = time.time() + 20
        while s.running and time.time() < deadline:       # leader exits by itself and is reaped by poll()
            time.sleep(0.05)
        victim = self.sentinel()                            # an unrelated new session leader ...
        s._pgid = victim.pid                                # ... that received the old PGID number (simulated PID reuse)
        s.stop(grace=1)
        time.sleep(0.3)
        self.assertTrue(alive(victim.pid), "stop() SIGKILLed an unrelated process group")
        self.assertNotIn("SIGKILL(group)", s.shutdown_report["signals"])
        self.assertTrue(s.shutdown_report.get("foreign_group_refused"))

    def test_orphaned_children_of_a_dead_leader_are_still_cleaned_up(self):
        s = self.srv("child"); s.start(); s.wait_ready(20, 0.1); time.sleep(0.5)
        pgid = s._pgid
        os.kill(s.pid, signal.SIGKILL); s._proc.wait(5)    # leader dies by itself; its engine-core-like child keeps the group alive
        self.assertTrue(s._group_alive())
        s.stop(grace=1)
        self.assertIn("SIGKILL(group)", s.shutdown_report["signals"]); self.assertTrue(s.shutdown_report["group_gone"])
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)


if __name__ == "__main__":
    unittest.main()
