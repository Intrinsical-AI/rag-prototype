#!/usr/bin/env python3
"""
Build and publish script for Intrinsical RAG Prototype.

This script automates the package building and publishing process for PyPI.
Cross-platform and without shell-specific commands.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    return result


def clean_build():
    """Clean previous build artifacts (cross-platform)."""
    print("🧹 Cleaning build artifacts...")

    for pattern in ("build", "dist", "*.egg-info"):
        for p in Path(".").glob(pattern):
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    # Python 3.8+: missing_ok supported
                    p.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception as e:
                print(f"⚠️  Warning: could not remove {p}: {e}")

    print("✅ Build artifacts cleaned")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    result = run_command("python -m pytest tests/ -v", check=False)
    
    if result.returncode != 0:
        print("❌ Tests failed. Please fix before building.")
        sys.exit(1)
    
    print("✅ All tests passed")


def run_linting():
    """Run code quality checks."""
    print("🔍 Running code quality checks...")
    
    # Run ruff
    result = run_command("ruff check src/", check=False)
    if result.returncode != 0:
        print("❌ Ruff linting failed. Please fix issues.")
        sys.exit(1)
    
    # Run black check
    result = run_command("black --check src/", check=False)
    if result.returncode != 0:
        print("❌ Black formatting check failed. Run 'black src/' to fix.")
        sys.exit(1)
    
    # Run mypy
    result = run_command("mypy src/", check=False)
    if result.returncode != 0:
        print("⚠️  MyPy found type issues (continuing anyway)")
    
    print("✅ Code quality checks passed")


def build_package():
    """Build the package."""
    print("📦 Building package...")

    # Install build dependencies
    run_command("python -m pip install --upgrade build twine")

    # Build the package
    run_command("python -m build")

    # Check the package (expand globs in Python for cross-platform compatibility)
    dist_files = [str(p) for p in Path("dist").glob("*")]
    if not dist_files:
        print("❌ No build artifacts found in 'dist/'. Did the build succeed?")
        sys.exit(1)
    run_command("twine check " + " ".join(dist_files))

    print("✅ Package built successfully")


def publish_to_test_pypi():
    """Publish to Test PyPI."""
    print("🚀 Publishing to Test PyPI...")

    dist_files = [str(p) for p in Path("dist").glob("*")]
    if not dist_files:
        print("❌ No build artifacts found in 'dist/'. Build first.")
        return

    result = run_command(
        "twine upload --repository testpypi " + " ".join(dist_files),
        check=False,
    )

    if result.returncode == 0:
        print("✅ Successfully published to Test PyPI")
        print("📝 Test installation with:")
        print("   pip install --index-url https://test.pypi.org/simple/ intrinsical-rag-prototype")
    else:
        print("❌ Failed to publish to Test PyPI")
        print(result.stderr)


def publish_to_pypi():
    """Publish to PyPI."""
    print("🚀 Publishing to PyPI...")

    # Confirm with user
    response = input("Are you sure you want to publish to PyPI? (yes/no): ")
    if response.lower() != "yes":
        print("❌ Publication cancelled")
        return

    dist_files = [str(p) for p in Path("dist").glob("*")]
    if not dist_files:
        print("❌ No build artifacts found in 'dist/'. Build first.")
        return

    result = run_command("twine upload " + " ".join(dist_files), check=False)

    if result.returncode == 0:
        print("✅ Successfully published to PyPI")
        print("📝 Install with: pip install intrinsical-rag-prototype")
    else:
        print("❌ Failed to publish to PyPI")
        print(result.stderr)


def main():
    """Main build and publish workflow."""
    if len(sys.argv) < 2:
        print("Usage: python build_package.py <command>")
        print("Commands:")
        print("  clean     - Clean build artifacts")
        print("  test      - Run tests")
        print("  lint      - Run code quality checks")
        print("  build     - Build package")
        print("  test-pypi - Publish to Test PyPI")
        print("  pypi      - Publish to PyPI")
        print("  all       - Run all steps (clean, test, lint, build)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "clean":
        clean_build()
    elif command == "test":
        run_tests()
    elif command == "lint":
        run_linting()
    elif command == "build":
        build_package()
    elif command == "test-pypi":
        publish_to_test_pypi()
    elif command == "pypi":
        publish_to_pypi()
    elif command == "all":
        clean_build()
        run_tests()
        run_linting()
        build_package()
        print("\n🎉 Package ready for publication!")
        print("Run 'python scripts/build_package.py test-pypi' to test publish")
        print("Run 'python scripts/build_package.py pypi' to publish to PyPI")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
