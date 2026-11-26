"""Bounded METADATA compatibility ranges (requirements/metadata-compat.txt), validated against the locked exact pins.

A range may only replace an exact lock pin `name==V` in the wheel's Requires-Dist when it is exactly `name>=V,<X.(Y+1)` for V = X.Y.Z: the lower bound is
the locked (tested, hash-pinned) version and the range never leaves that major.minor series. Anything else is rejected, so no unbounded or
cross-series relaxation can enter a wheel."""
from __future__ import annotations

import pathlib
import re

from packaging.requirements import Requirement
from packaging.version import Version


def canon(s: str) -> str:
    return re.sub(r"[-_.]+", "-", s).lower()


def parse(text: str, pins: dict[str, str]) -> dict[str, str]:
    """{canonical name: 'name<Y,>=X'} for every compat line; ``pins`` maps canonical names to the locked exact versions. Raises ValueError on any violation."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        ln = raw.split("#")[0].strip()
        if not ln:
            continue
        req = Requirement(ln)
        name = canon(req.name)
        specs = {s.operator: Version(s.version) for s in req.specifier}
        if req.marker or req.extras or req.url or len(req.specifier) != 2 or set(specs) != {">=", "<"}:
            raise ValueError(f"compat range must be exactly 'name>=X,<Y': {ln}")
        if name not in pins:
            raise ValueError(f"{name} is not pinned by any lock; only a locked exact pin may be relaxed")
        lo, hi, locked = specs[">="], specs["<"], Version(pins[name])
        if lo != locked:
            raise ValueError(f"{name}: lower bound {lo} must equal the locked version {locked}")
        if hi.release[:2] != (lo.major, lo.minor + 1) or hi.release[2:] not in ((), (0,)) or hi.pre or hi.dev or hi.post:
            raise ValueError(f"{name}: upper bound {hi} must close the {lo.major}.{lo.minor} series (expected <{lo.major}.{lo.minor + 1})")
        if name in out:
            raise ValueError(f"{name} is listed twice")
        out[name] = str(req)
    return out


def load(path: pathlib.Path, pins: dict[str, str]) -> dict[str, str]:
    return parse(path.read_text(), pins) if path.exists() else {}
