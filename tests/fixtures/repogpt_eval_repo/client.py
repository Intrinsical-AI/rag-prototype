class ApiClient:
    """Tiny API client used for retrieval demos."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def build_request(self, path: str) -> str:
        """Build an API request URL for a resource path."""
        return f"{self.base_url}/{path.lstrip('/')}"


def build_api_client(base_url: str) -> ApiClient:
    """Build the API client configured for a base URL."""
    return ApiClient(base_url)
