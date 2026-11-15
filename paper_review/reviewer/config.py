import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path):
    """KEY=value lines from .env, which keeps addresses and keys out of the code; real environment variables win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and not line.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


load_env_file(ROOT / ".env")
ORCHESTRATOR = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080")

MODEL_PROFILES = {
    # Swift 1.5 27B as Strap runs it: the build that fits whole on the GPU with its multi-token-prediction head
    # (92-100 tokens/s), two slots, its card's sampling. Agents think at medium, the template's neutral effort;
    # one-shot answers (extraction, screening, digests, verdicts) skip thinking, as Strap's helper calls do.
    "swift-1.5-27b-xxs": {
        "slots": 2,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 20, "min_p": 0.0}},
        "agent_effort": "medium",
        "answer_effort": "none",
    },
}
DEFAULT_PROFILE = {"slots": 1, "sampling": {"temperature": 0.7}, "agent_effort": None, "answer_effort": None}


@dataclass
class Settings:
    model: str = os.environ.get("LLM_MODEL", "swift-1.5-27b-xxs")
    llm_url: str = os.environ.get("LLM_BASE_URL", f"{ORCHESTRATOR}/api/llm/v1")
    api_key: str = os.environ.get("LLM_API_KEY", "not-needed")
    embeddings_url: str = f"{ORCHESTRATOR}/api/embeddings"
    search_url: str = f"{ORCHESTRATOR}/api/search"
    cache_dir: Path = ROOT / ".cache"
    protocol: dict = field(default_factory=dict)

    @property
    def profile(self) -> dict:
        return MODEL_PROFILES.get(self.model, DEFAULT_PROFILE)


def read_protocol(path: Path) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def load_settings(protocol_path: Path = ROOT / "protocol.toml", **overrides) -> Settings:
    loaded = Settings(**overrides)
    loaded.protocol = read_protocol(protocol_path)
    loaded.cache_dir.mkdir(exist_ok=True)
    return loaded


settings = load_settings()
