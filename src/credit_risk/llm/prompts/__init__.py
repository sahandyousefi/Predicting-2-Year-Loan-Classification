from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

PROMPT_DIR = Path(__file__).parent


STATUSES = ("production", "candidate")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str
    status: str = "production"

    @property
    def fingerprint(self) -> str:
        """Hash of the template text, logged with every generation."""
        return hashlib.sha256((self.system + self.user).encode()).hexdigest()[:12]

    @property
    def variables(self) -> set[str]:
        return set(re.findall(r"\$(\w+)", self.user))

    def render(self, **values: object) -> tuple[str, str]:
        missing = self.variables - set(values)
        if missing:
            raise KeyError(f"missing prompt variables: {sorted(missing)}")
        return self.system.strip(), Template(self.user).substitute(values).strip()


def available_versions(name: str) -> list[int]:
    return sorted(
        int(p.stem.rsplit("_v", 1)[1]) for p in PROMPT_DIR.glob(f"{name}_v*.yaml")
    )


def _read(name: str, version: int) -> PromptTemplate:
    path = PROMPT_DIR / f"{name}_v{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no version {version} of prompt '{name}'")
    raw = yaml.safe_load(path.read_text())
    status = raw.get("status", "production")
    if status not in STATUSES:
        raise ValueError(f"unknown prompt status '{status}' in {path.name}")
    return PromptTemplate(
        raw["name"], int(raw["version"]), raw["system"], raw["user"], status
    )


def production_version(name: str) -> int:
    """Latest version of ``name`` whose status is production."""
    versions = available_versions(name)
    if not versions:
        raise FileNotFoundError(f"no prompt named '{name}'")
    production = [v for v in versions if _read(name, v).status == "production"]
    if not production:
        raise FileNotFoundError(f"prompt '{name}' has no production version")
    return production[-1]


def load_prompt(name: str, version: int | None = None) -> PromptTemplate:
    """Load a prompt by name.

    Without ``version`` this is the latest production version. Candidate
    versions are loaded only when requested, so a new prompt reaches
    production by changing its status after the comparison report, not by
    being added.
    """
    return _read(name, version or production_version(name))
