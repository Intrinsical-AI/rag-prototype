# Changelog

All notable changes to the Intrinsical RAG Prototype will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready packaging configuration for PyPI distribution
- Enhanced CLI with Click framework and better UX
- Pre-commit hooks for code quality assurance
- Comprehensive environment configuration example
- Professional .gitignore optimized for RAG projects
- Packaged frontend (`local_rag_backend/frontend/index.html`) served automatically via FastAPI
- Packaged sample data (`local_rag_backend/data/faq.csv`) with fallback loading in scripts

### Changed
- Improved pyproject.toml with better metadata and classifiers
- Enhanced dependency version constraints for stability
- Optimized package structure for distribution
- Cross-platform build/publish script (`scripts/build_package.py`) without shell-specific commands
- Unified pytest configuration in `pyproject.toml`; minimized `pytest.ini`
- Minimized `setup.cfg` to avoid duplication with `pyproject.toml`
- Trimmed `MANIFEST.in` to include only necessary files (removed `pytest.ini`, `.pre-commit-config.yaml`)
- Minor README polish (removed `make` dependency in test commands)
- Adopted src-layout packaging with `package-dir = {"" = "src"}`
- Fixed console script entry points to `local_rag_backend.cli:*`
- Uvicorn import path corrected to `local_rag_backend.app.main:app`
- README updated to reflect new module paths and packaged frontend behavior
- `.env.example` now documents `OPENAI_TOP_P`

### Removed
- `setuptools-scm` from build-system to simplify versioning (manual `project.version`)
## [0.1.0] - 2025-01-06

### Added
- Initial release of Intrinsical RAG Prototype
- Hexagonal architecture implementation with ports & adapters
- Multiple retrieval modes: sparse (BM25), dense (FAISS), hybrid
- FastAPI backend with async support
- Support for OpenAI and Ollama LLM providers
- SQLite database with SQLAlchemy ORM
- Comprehensive test suite (unit, integration, e2e)
- Docker containerization with multi-stage builds
- CI/CD pipeline with GitHub Actions
- Bootstrap and index building scripts
- Vanilla HTML/CSS/JS frontend
- Production-ready configuration management

### Features
- Document ingestion from CSV files
- Vector search with FAISS integration
- Sparse search with BM25 ranking
- Hybrid retrieval combining multiple approaches
- RESTful API with automatic documentation
- Environment-based configuration
- Comprehensive logging and error handling
- Database migration support
- CLI tools for common operations

### Documentation
- Comprehensive README with architecture diagrams
- API documentation with OpenAPI/Swagger
- Development setup instructions
- Docker deployment guides
- Architecture documentation with hexagonal pattern explanation

[Unreleased]: https://github.com/Intrinsical-AI/rag-prototype/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Intrinsical-AI/rag-prototype/releases/tag/v0.1.0
