from collections.abc import Mapping


def parse_timeout(raw: str | None) -> int:
    """Parse timeout values from configuration strings."""
    if raw is None or not raw.strip():
        return 30
    return max(1, int(raw))


def load_config(env: Mapping[str, str]) -> dict[str, int | str]:
    """Load configuration values from environment mappings."""
    return {
        "api_url": env.get("API_URL", "https://api.example.com"),
        "timeout_s": parse_timeout(env.get("TIMEOUT_S")),
    }
