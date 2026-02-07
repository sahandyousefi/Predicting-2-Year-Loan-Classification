from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str

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


def load_prompt(name: str, version: int | None = None) -> PromptTemplate:
    """Load a prompt by name; the latest version unless one is given."""
    versions = available_versions(name)
    if not versions:
        raise FileNotFoundError(f"no prompt named '{name}'")
    version = version or versions[-1]
    raw = yaml.safe_load((PROMPT_DIR / f"{name}_v{version}.yaml").read_text())
    return PromptTemplate(raw["name"], int(raw["version"]), raw["system"], raw["user"])
