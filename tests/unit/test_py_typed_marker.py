# tests/unit/test_py_typed_marker.py
import importlib.resources as resources


def test_package_has_py_typed_marker():
    marker = resources.files("local_rag_backend").joinpath("py.typed")
    assert marker.is_file()

