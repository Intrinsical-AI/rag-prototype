class Demo:
    """Demo class for simple method retrieval."""

    def method(self, value: int) -> int:
        """Increment a value by one."""
        return value + 1


def helper(name: str = "world") -> str:
    """Return a friendly greeting helper string."""
    return f"hello {name}"
