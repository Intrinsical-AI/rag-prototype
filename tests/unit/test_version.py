from importlib.metadata import version


def test_package_version_matches_dist_metadata():
    import local_rag_backend

    assert local_rag_backend.__version__ == version("rag-prototype")
    assert local_rag_backend.__version__ != "0.0.0"
