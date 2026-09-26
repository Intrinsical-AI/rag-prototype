def extract_bearer_token(auth_header: str | None) -> str | None:
    """Extract a bearer token from an Authorization header."""
    if not auth_header:
        return None
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def validate_token(token: str) -> bool:
    """Validate an auth token using a signed prefix and minimum length."""
    cleaned = token.strip()
    return cleaned.startswith("signed-") and len(cleaned) >= 12
