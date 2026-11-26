"""ServerProcess must only ever signal ITS OWN server's process group (CPU-only, fake server; no vLLM, torch, CUDA or NVML).

Background (Enigma x Kaggle fatal-session investigation, 2026-11-26): every Kaggle process runs as root, so a signal sent to the wrong process group
is never refused by the kernel. After the server's leader has exited and been reaped, its PGID number can be reused by an unrelated new session leader
(any ``start_new_session=True`` child, e.g. the Enigma connector). Linux never hands out a PID that is still in use as a group id, so: once our leader is
reaped, a live process whose PID equals our PGID is NOT ours, while our own orphaned descendants never have that PID."""
import os, pathlib, signal, subprocess, sys, threading, time, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from test_serving_runtime import Fake          # noqa: E402  (reuses the fake interpreter / fake server)
from stokastisk import serving                 # noqa: E402


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
        self.assertIn("SIGKILL(orphans=1)", s.shutdown_report["signals"]); self.assertTrue(s.shutdown_report["group_gone"])
        self.assertEqual(s.shutdown_report["orphan_signals_refused"], 0)
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)


class OrphanCleanupIsRaceFree(Fake):
    """After the leader is reaped, orphans are signalled one by one through a pidfd, only after re-verifying group AND session; killpg is never used."""

    def sentinel(self):
        s = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"], start_new_session=True)
        self.addCleanup(lambda: (s.kill(), s.wait(5)))
        return s

    def test_a_process_holding_the_pgid_number_is_never_signalled(self):
        victim = self.sentinel()                          # its group and session ARE (victim.pid, victim.pid): exactly what a reused PGID looks like
        s = serving.ServerProcess.__new__(serving.ServerProcess)
        with mock.patch.object(serving, "_group_members", return_value=[victim.pid]):
            done, refused = s._kill_orphans(victim.pid)
        time.sleep(0.2)
        self.assertEqual((done, refused), (0, 1)); self.assertTrue(alive(victim.pid))

    def test_identity_is_rechecked_after_pinning_and_a_mismatch_is_refused(self):
        victim = self.sentinel()
        s = serving.ServerProcess.__new__(serving.ServerProcess)
        with mock.patch.object(serving, "_group_members", return_value=[victim.pid]), mock.patch.object(serving, "_proc_ids", return_value=(1, 1)):
            done, refused = s._kill_orphans(4242424)
        time.sleep(0.2)
        self.assertEqual((done, refused), (0, 1)); self.assertTrue(alive(victim.pid))

    def test_without_pidfd_nothing_is_signalled(self):
        victim = self.sentinel()
        s = serving.ServerProcess.__new__(serving.ServerProcess)
        with mock.patch.object(serving.os, "pidfd_open", None), mock.patch.object(serving, "_group_members", return_value=[victim.pid]):
            self.assertEqual(s._kill_orphans(4242424), (0, 1))
        self.assertTrue(alive(victim.pid))

    def test_permission_error_counts_as_refused(self):
        victim = self.sentinel()
        s = serving.ServerProcess.__new__(serving.ServerProcess)
        pg = os.getpgid(victim.pid)
        with mock.patch.object(serving, "_group_members", return_value=[victim.pid]), mock.patch.object(serving, "_proc_ids", return_value=(pg + 1, pg + 1)), \
                mock.patch.object(serving.signal, "pidfd_send_signal", side_effect=PermissionError):
            self.assertEqual(s._kill_orphans(pg + 1), (0, 1))
        self.assertTrue(alive(victim.pid))

    def test_group_members_requires_both_group_and_session(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])          # same session as the test, own group? no: inherits ours
        self.addCleanup(lambda: (child.kill(), child.wait(5)))
        self.assertNotIn(child.pid, serving._group_members(child.pid))                           # its group/session are not (child.pid, child.pid)


class LifecycleConcurrency(Fake):
    def test_concurrent_stops_signal_once_and_both_return_after_shutdown(self):
        s = self.srv(); s.start(); s.wait_ready(20, 0.1)
        results = []
        ts = [threading.Thread(target=lambda: results.append(s.stop(grace=10))) for _ in range(3)]
        for t in ts: t.start()
        for t in ts: t.join(60)
        self.assertEqual(sorted(results, key=lambda r: r is None), [0, None, None])
        self.assertEqual(s.shutdown_report["signals"], ["SIGINT"]); self.assertFalse(s.running)

    def test_concurrent_start_and_stop_never_leave_a_server_running(self):
        for _ in range(3):
            s = self.srv()
            t = threading.Thread(target=s.start); t.start()
            s.stop(grace=10); t.join(30)
            s.stop(grace=10)                                     # whichever order the lock chose, a final stop() leaves nothing behind
            self.assertFalse(s.running); self.assertIsNone(s._pgid)

    def test_failed_startup_closes_the_log_handle(self):
        s = serving.ServerProcess("m", self.cfg, python="/nonexistent/python", log_path=str(self.td / "x.log"), port=serving.find_free_port())
        with self.assertRaises(Exception):
            s.start()
        self.assertIsNone(s._log); self.assertIsNone(s.stop())


if __name__ == "__main__":
    unittest.main()
