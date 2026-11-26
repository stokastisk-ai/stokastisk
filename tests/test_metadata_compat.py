"""requirements/metadata-compat.txt: only narrowly bounded, evidence-backed relaxations of locked pins can reach a wheel's Requires-Dist (CPU only)."""
import pathlib, re, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "release"))
import metadata_compat as mc  # noqa: E402

PINS = {"pydantic-settings": "2.16.2", "supervisor": "4.3.0"}


def locked_pins():
    out = {}
    for lock in (ROOT / "requirements").glob("*.lock"):
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==(\S+) \\$", lock.read_text(), re.M):
            out[mc.canon(m.group(1))] = m.group(2)
    return out


class MetadataCompat(unittest.TestCase):
    def test_the_repository_file_is_valid_and_only_relaxes_pydantic_settings_within_2_16(self):
        got = mc.load(ROOT / "requirements" / "metadata-compat.txt", locked_pins())
        self.assertEqual(got, {"pydantic-settings": "pydantic-settings<2.17,>=2.16.2"})

    def test_the_kaggle_version_satisfies_it_and_the_locked_version_still_does(self):
        from packaging.requirements import Requirement
        spec = Requirement(mc.load(ROOT / "requirements" / "metadata-compat.txt", locked_pins())["pydantic-settings"]).specifier
        self.assertIn("2.16.3", spec); self.assertIn("2.16.2", spec); self.assertNotIn("2.17.0", spec); self.assertNotIn("2.16.1", spec)

    def test_the_lock_still_hash_pins_the_tested_version(self):
        self.assertEqual(locked_pins()["pydantic-settings"], "2.16.2")

    def test_rejections(self):
        bad = ["pydantic-settings>=2.16.2",                 # unbounded
               "pydantic-settings>=2.16.2,<3",               # crosses minor series
               "pydantic-settings>=2.16.0,<2.17",            # lower bound below the locked version
               "pydantic-settings>=2.16.3,<2.17",            # lower bound above the locked (tested) version
               "pydantic-settings~=2.16.2",                  # other operator
               "pydantic-settings>=2.16.2,<2.17; python_version<'3.14'",
               "unlocked-pkg>=1.0,<1.1",                    # not pinned by a lock
               "pydantic-settings>=2.16.2,<2.17\npydantic_settings>=2.16.2,<2.17"]
        for b in bad:
            with self.subTest(b), self.assertRaises(ValueError):
                mc.parse(b, PINS)

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertEqual(mc.parse("# x\n\n  # y\n", PINS), {})


if __name__ == "__main__":
    unittest.main()
