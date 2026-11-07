"""Accept RAG's retained wheel/sdist; reuse installed embedding and snapshot probes."""

from __future__ import annotations

import argparse
import configparser
import contextlib
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import re
import runpy
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zipfile import ZipFile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard() -> dict[str, Any]:
    denied: list[str] = []

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        internet = event == "socket.__new__" and arguments[1] in {socket.AF_INET, socket.AF_INET6}
        if internet or event in {
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
            "socket.getnameinfo",
        }:
            denied.append(event)
            raise PermissionError("Package acceptance denies Internet sockets and DNS")

    sys.addaudithook(audit)
    for operation in (
        lambda: socket.socket(socket.AF_INET),
        lambda: socket.socket(socket.AF_INET6),
        lambda: socket.getaddrinfo("package.invalid", 443),
    ):
        try:
            operation()
        except PermissionError:
            pass
        else:
            raise AssertionError("Network audit self-test failed")
    left, right = socket.socketpair()
    left.close()
    right.close()
    selftest = list(denied)
    assert len(selftest) == 3
    denied.clear()
    return {"selftest_denials": selftest, "denied_operations": denied, "unix_allowed": True}


def helper(name: str) -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name(name)))


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=240)
    with log.open("a") as stream:
        stream.write(json.dumps(command) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")
    return result.stdout


def requirement(value: str) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    """Normalize only this project's names, extras, specifiers and extra markers."""
    head, _, marker = value.partition(";")
    match = re.fullmatch(
        r"\s*([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[([A-Za-z0-9_,.-]+)\])?\s*(.*?)\s*", head
    )
    assert match is not None, value

    def normalize(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    specs = match[3].strip().removeprefix("(").removesuffix(")")
    normalized_marker = ""
    if marker:
        extra = re.fullmatch(r"\s*extra\s*==\s*(['\"])([A-Za-z0-9_.-]+)\1\s*", marker)
        assert extra is not None, "Review new conditional markers explicitly"
        normalized_marker = normalize(extra[2])
    return (
        normalize(match[1]),
        tuple(sorted(normalize(name) for name in (match[2] or "").split(",") if name)),
        tuple(sorted(part.replace(" ", "") for part in specs.split(",") if part)),
        normalized_marker,
    )


def check_metadata(raw: bytes, config: dict[str, Any], readme: bytes) -> None:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    project = config["project"]
    assert message["Name"] == project["name"] == "rag-prototype"
    assert message["Version"] == project["version"]
    assert (
        requirement("python" + message["Requires-Python"])[2]
        == requirement("python" + project["requires-python"])[2]
    )
    assert message["License-Expression"] == project["license"]
    assert message.get_all("License-File") == ["LICENSE"]
    assert message.get_all("Provides-Extra") == list(project["optional-dependencies"])
    expected = list(project["dependencies"])
    for extra, dependencies in project["optional-dependencies"].items():
        expected.extend(value + '; extra == "' + extra + '"' for value in dependencies)
    actual = message.get_all("Requires-Dist") or []
    assert len(actual) == len(expected)
    assert {requirement(value) for value in actual} == {requirement(value) for value in expected}
    description = message.get_payload(decode=True)
    assert isinstance(description, bytes) and description == readme, (
        "README metadata differs from source"
    )


def check_pair(wheel: Path, sdist: Path, package: dict[str, str], inputs: dict[str, bytes]) -> str:
    """Bind runtime bytes and distributable metadata/documentation to Git-owned inputs."""
    config = tomllib.loads(inputs["pyproject.toml"].decode())
    version = config["project"]["version"]
    assert wheel.name == f"rag_prototype-{version}-py3-none-any.whl"
    assert sdist.name == f"rag_prototype-{version}.tar.gz"
    prefix = f"rag_prototype-{version}.dist-info/"
    runtime = {name.removeprefix("src/"): digest for name, digest in package.items()}
    generated = {
        prefix + name
        for name in (
            "licenses/LICENSE",
            "METADATA",
            "WHEEL",
            "entry_points.txt",
            "top_level.txt",
            "RECORD",
        )
    }
    with ZipFile(wheel) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert all(name in runtime or name in generated for name in archive.namelist())
        assert all(not stat.S_ISLNK(item.external_attr >> 16) for item in archive.infolist())
        assert {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("local_rag_backend/")
        } == runtime
        check_metadata(archive.read(prefix + "METADATA"), config, inputs["README.md"])
        assert archive.read(prefix + "licenses/LICENSE") == inputs["LICENSE"]
        entrypoints = configparser.ConfigParser()
        entrypoints_raw = archive.read(prefix + "entry_points.txt")
        entrypoints.read_string(entrypoints_raw.decode())
        assert dict(entrypoints["console_scripts"]) == config["project"]["scripts"]
        wheel_headers = archive.read(prefix + "WHEEL").decode()
    generated_sdist = {
        "src/rag_prototype.egg-info/" + name
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    found: dict[str, bytes] = {}
    with tarfile.open(sdist) as archive:
        seen: set[str] = set()
        for member in archive:
            path = Path(member.name)
            assert member.name not in seen and not path.is_absolute() and ".." not in path.parts
            seen.add(member.name)
            assert path.parts[0] == f"rag_prototype-{version}"
            assert member.isdir() or member.isfile()
            if member.isfile():
                name = Path(*path.parts[1:]).as_posix()
                assert (
                    name in package
                    or name in inputs
                    or name in {"PKG-INFO", "setup.cfg"} | generated_sdist
                )
                content = archive.extractfile(member)
                assert content is not None
                found[name] = content.read()
    assert all(
        hashlib.sha256(found[name]).hexdigest() == digest for name, digest in package.items()
    )
    assert all(found[name] == content for name, content in inputs.items()), (
        "Sdist source input differs from Git"
    )
    for name in ("PKG-INFO", "src/rag_prototype.egg-info/PKG-INFO"):
        check_metadata(found[name], config, inputs["README.md"])
    assert found["src/rag_prototype.egg-info/entry_points.txt"] == entrypoints_raw
    return wheel_headers


def launch(console: Path, output: Path, readonly: bool, arguments: list[str]) -> None:
    audit = guard()
    assert console.parent == Path(sys.executable).parent and console.is_file()
    import httpx

    # Only the external provider's HTTP boundary is deterministic. Installed RAG,
    # snapshot/retrieval, official SDK and stdio lifecycle remain unmodified.
    calls: list[str] = []

    def provider(url: str, **kwargs: Any) -> httpx.Response:
        assert url == "http://provider.invalid/api/generate" and kwargs["json"]["stream"] is False
        calls.append(url)
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"response": "synthetic answer"}
        )

    provider_patch = patch.object(httpx, "post", provider)
    provider_patch.start()
    writes: list[str] = []

    def stop() -> None:
        pass

    if readonly:
        from local_rag_backend import mcp_server

        assert Path(mcp_server.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        writes, stop = helper("smoke_ephemeral_snapshot_wheel.py")["_read_audit"]()
    try:
        sys.argv = [str(console), *arguments]
        runpy.run_path(str(console), run_name="__main__")
    finally:
        stop()
        provider_patch.stop()
        output.write_text(
            json.dumps(
                {
                    "network": audit,
                    "write_attempts": writes,
                    "provider_calls": calls,
                    "provider_boundary": "httpx.post deterministic stub; no model/provider acceptance",
                    "console": str(console),
                    "console_sha256": sha(console),
                    "pid": os.getpid(),
                    "completed": True,
                }
            )
        )
        assert not writes and not audit["denied_operations"]


def installed(wheel: Path, artifact: Path) -> dict[str, Any]:
    import local_rag_backend

    distribution = metadata.distribution("rag-prototype")
    package = Path(local_rag_backend.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve())
    direct_url = distribution.read_text("direct_url.json")
    assert direct_url is not None and json.loads(direct_url)["url"] == artifact.as_uri()
    with ZipFile(wheel) as archive:
        expected = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("local_rag_backend/") and not name.endswith("/")
        }
        for member in archive.namelist():
            if member.endswith(("/METADATA", "/entry_points.txt", "/licenses/LICENSE")):
                actual = Path(str(distribution.locate_file(member))).resolve()
                assert actual.is_relative_to(Path(sys.prefix).resolve())
                assert actual.read_bytes() == archive.read(member), (
                    "Installed metadata/license differs from verified wheel"
                )
        message = BytesParser().parsebytes(
            archive.read(next(name for name in archive.namelist() if name.endswith("/METADATA")))
        )
        assert distribution.version == message["Version"]
    assert {
        path.relative_to(package.parent).as_posix()
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    } == set(expected)
    assert all((package.parent / name).read_bytes() == value for name, value in expected.items())
    assert all(
        importlib.util.find_spec(name) is None
        for name in (
            "repogpt",
            "chat_exports",
            "tr3v0r_app",
            "tr3v0r_core",
            "tr3v0r_parsers",
            "tr3v0r_analytics",
            "tr3v0r_sensors",
        )
    )
    return {
        "version": distribution.version,
        "origin": str(package),
        "environment": str(Path(sys.prefix).resolve()),
        "python": sys.executable,
        "runtime_files": {
            name: hashlib.sha256(value).hexdigest() for name, value in expected.items()
        },
        "metadata": distribution.read_text("METADATA"),
        "entry_points": distribution.read_text("entry_points.txt"),
    }


async def sdk(work: Path, env: dict[str, str]) -> dict[str, Any]:
    from jsonschema import Draft202012Validator
    from mcp import ClientSession, McpError, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    audit_file = work / "server-audit.json"
    parameters = StdioServerParameters(
        command=sys.executable,
        cwd=str(work),
        env=env,
        args=[
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--launch",
            str(Path(sys.executable).with_name("rag-mcp")),
            str(audit_file),
            "readonly",
        ],
    )
    async with (
        stdio_client(parameters) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        initialized = await session.initialize()
        assert initialized.serverInfo.name == "rag-prototype"
        assert initialized.protocolVersion == "2025-11-25"
        await session.send_ping()
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}

        async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            result = await session.call_tool(name, arguments)
            assert not result.isError and result.structuredContent
            assert isinstance(result.content[0], types.TextContent)
            assert json.loads(result.content[0].text) == result.structuredContent
            Draft202012Validator(tools[name].outputSchema).validate(result.structuredContent)
            return result.structuredContent

        status = await call("rag_status", {})
        assert status["health"]["snapshot_ready"] is True
        query = await call("rag_ask_ephemeral", {"question": "alpha", "k": 1})
        assert (
            query["answer"] == "synthetic answer" and query["sources"][0]["document"]["id"] == "one"
        )
        assert query["snapshot_id"] == status["health"]["snapshot_id"]
        try:
            await session.call_tool("rag_ask_ephemeral", {"question": "alpha", "k": True})
        except McpError as error:
            assert error.error.code == -32602
        else:
            raise AssertionError("Strict SDK arguments were accepted")
    server = json.loads(audit_file.read_text())
    assert (
        server["completed"]
        and not server["write_attempts"]
        and not server["network"]["denied_operations"]
    )
    assert len(server["provider_calls"]) == 1
    try:
        os.kill(server["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("SDK server was not reaped")
    return {
        "protocol": initialized.protocolVersion,
        "tools": sorted(tools),
        "status": status,
        "query": query,
        "strict_invalid_arguments": 1,
        "server": server,
        "server_reaped": True,
    }


def embedding_probe(consumer: Path, arguments: list[str]) -> None:
    audit = guard()
    sys.argv = [str(consumer), *arguments]
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        runpy.run_path(str(consumer), run_name="__main__")
    assert not audit["denied_operations"]
    print(
        json.dumps(
            {
                "payload": json.loads(stream.getvalue()),
                "network": audit,
                "explicit_arguments": arguments,
                "config_environment": os.environ.get("RAG_CONFIG_PATH"),
                "pid": os.getpid(),
            }
        )
    )


def probe(wheel: Path, artifact: Path, output: Path) -> None:
    audit = guard()
    receipt = installed(wheel, artifact)
    embedding = helper("smoke_embedding_api_wheel.py")
    snapshot = helper("smoke_ephemeral_snapshot_wheel.py")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    work = output.parent
    consumer = work / "embedding-consumer.py"
    consumer.write_text(embedding["CONSUMER_SCRIPT"])
    selections: dict[str, Any] = {}
    for selection in ("explicit", "environment"):
        directory = work / f"embedding-{selection}"
        directory.mkdir()
        configuration = directory / "config.yaml"
        configuration.write_text(
            embedding["SYNTHETIC_CONFIG"]
            .replace("data/cache.sqlite3", f"data-{selection}/cache.sqlite3")
            .replace("data_dir: data\n", f"data_dir: data-{selection}\n")
        )
        selected = embedding["_clean_process_env"]()
        arguments = [str(configuration)] if selection == "explicit" else []
        if selection == "environment":
            selected["RAG_CONFIG_PATH"] = str(configuration)
        completed = run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--embedding",
                str(consumer),
                *arguments,
            ],
            cwd=work,
            env=selected,
            log=work / "embedding.log",
        )
        selections[selection] = json.loads(completed)
        payload = selections[selection]["payload"]
        embedding["_assert_payload"](payload, venv_dir=Path(sys.prefix))
        assert (
            Path(payload["status"]["cache_db_path"])
            == directory / f"data-{selection}/cache.sqlite3"
        )
        assert selections[selection]["explicit_arguments"] == arguments
        assert selections[selection]["config_environment"] == selected.get("RAG_CONFIG_PATH")
    assert selections["explicit"]["config_environment"] is None
    assert selections["explicit"]["pid"] != selections["environment"]["pid"]
    assert (
        selections["explicit"]["payload"]["vectors"]
        == selections["environment"]["payload"]["vectors"]
    )
    fixture = work / "fixture"
    fixture.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        snapshot["_fixture"](fixture, "http://provider.invalid")
    cli_receipts: list[dict[str, Any]] = []
    for mode in ("sparse", "dense", "dual", "hybrid"):
        case = fixture / mode
        selected = {**env, "RAG_CONFIG_PATH": str(case / "config.json")}
        for console, arguments in (
            ("rag-status", []),
            ("rag-snapshot", [str(case / "frozen")]),
            ("rag-status", ["--help"]),
        ):
            destination = work / f"{mode}-{console}-{len(cli_receipts)}.json"
            run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--launch",
                    str(Path(sys.executable).with_name(console)),
                    str(destination),
                    "writer",
                    *arguments,
                ],
                cwd=work,
                env=selected,
                log=work / "console.log",
            )
            cli_receipts.append(json.loads(destination.read_text()))
        assert b"PRIVATE-HISTORY-SYNTHETIC" not in (case / "frozen/documents.sqlite3").read_bytes()
    before = snapshot["_digests"](fixture)
    import anyio

    receipt["sdk"] = {}
    for mode in ("sparse", "dense", "dual", "hybrid"):
        selected = {**env, "RAG_CONFIG_PATH": str(fixture / mode / "config.json")}
        mode_work = work / f"sdk-{mode}"
        mode_work.mkdir()
        receipt["sdk"][mode] = anyio.run(sdk, mode_work, selected)
    assert snapshot["_digests"](fixture) == before and not audit["denied_operations"]
    receipt.update(
        status="passed",
        network=audit,
        cli=cli_receipts,
        snapshot_inputs=before,
        snapshot_inputs_unchanged=True,
        embedding_selection=selections,
        cli_query="not_applicable; no query CLI exists",
        cli_status_phase="explicit writer setup",
    )
    output.write_text(json.dumps(receipt, indent=2) + "\n")


def accept(root: Path, dist: Path, output: Path) -> dict[str, Any]:
    canonical = runpy.run_path(str(root / "scripts/smoke_canonical_wheels.py"))
    embedding = runpy.run_path(str(root / "scripts/smoke_embedding_api_wheel.py"))
    env = embedding["_clean_process_env"]()
    env.update(PYTHONDONTWRITEBYTECODE="1", UV_PYTHON_DOWNLOADS="never")
    uv = shutil.which("uv")
    assert uv
    source = canonical["_consumer_source"](env)
    for name in (
        ".github/workflows/ci.yml",
        "README.md",
        "LICENSE",
        "scripts/smoke_package.py",
        "scripts/smoke_embedding_api_wheel.py",
        "scripts/smoke_ephemeral_snapshot_wheel.py",
    ):
        source["files"][name] = sha(root / name)
    source["checkout_path"] = str(root)
    if os.environ.get("CI_SOURCE_SHA"):
        assert source["revision"] == os.environ["CI_SOURCE_SHA"] and not source["dirty"]
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    wheel, sdist = wheels[0], sdists[0]
    assert wheel.name.startswith("rag_prototype-") and sdist.name.startswith("rag_prototype-")
    canonical["_verify_source_wheel"](wheel, source)
    package = {
        path: value
        for path, value in source["files"].items()
        if path.startswith("src/local_rag_backend/")
    }
    tracked = set(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--",
                "src/local_rag_backend",
                "README.md",
                "LICENSE",
                "pyproject.toml",
            ]
        )
        .decode()
        .split("\0")
    )
    assert set(package) == {name for name in tracked if name.startswith("src/local_rag_backend/")}
    assert {"README.md", "LICENSE", "pyproject.toml"} <= tracked
    inputs = {
        name: (root / name).read_bytes() for name in ("README.md", "LICENSE", "pyproject.toml")
    }
    wheel_headers = check_pair(wheel, sdist, package, inputs)
    artifacts = {path.name: sha(path) for path in (wheel, sdist)}
    (output / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in artifacts.items())
    )
    scratch = Path(os.environ.get("RUNNER_TEMP", str(root.parent))).resolve()
    assert not scratch.is_relative_to(root)
    work = Path(tempfile.mkdtemp(prefix="rag-package-", dir=scratch))
    for name in (
        "smoke_package.py",
        "smoke_embedding_api_wheel.py",
        "smoke_ephemeral_snapshot_wheel.py",
    ):
        shutil.copyfile(root / "scripts" / name, work / name)
    requirements = work / "requirements.txt"
    run(
        [
            uv,
            "export",
            "--frozen",
            "--no-default-groups",
            "--extra",
            "server",
            "--extra",
            "mcp",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        cwd=root,
        env=env,
        log=output / "commands.log",
    )
    receipt: dict[str, Any] = {
        "source": source,
        "artifacts": artifacts,
        "installed": {},
        "scratch": str(work),
        "requirements_sha256": sha(requirements),
        "python": sys.version,
        "uv": run([uv, "--version"], cwd=root, env=env, log=output / "commands.log").strip(),
        "wheel_headers": wheel_headers,
        "github": {
            key: env[key]
            for key in (
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_WORKFLOW_REF",
                "GITHUB_WORKFLOW_SHA",
            )
            if key in env
        },
    }
    for kind, artifact in (("wheel", wheel), ("sdist", sdist)):
        environment = work / f"venv-{kind}"
        run(
            [uv, "venv", "--python", sys.executable, str(environment)],
            cwd=work,
            env=env,
            log=output / "commands.log",
        )
        python = environment / "bin/python"
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            cwd=work,
            env=env,
            log=output / "commands.log",
        )
        run(
            [uv, "pip", "install", "--python", str(python), "--no-deps", str(artifact)],
            cwd=work,
            env=env,
            log=output / "commands.log",
        )
        case = work / kind
        case.mkdir()
        run(
            [
                str(python),
                "-I",
                "-B",
                str(work / "smoke_package.py"),
                "--probe",
                str(wheel),
                str(artifact),
                str(case / "receipt.json"),
            ],
            cwd=case,
            env=env,
            log=output / "commands.log",
        )
        receipt["installed"][kind] = json.loads((case / "receipt.json").read_text())
    assert all(sha(dist / name) == value for name, value in artifacts.items())
    assert all(sha(root / name) == value for name, value in source["files"].items())
    shutil.rmtree(work)
    receipt.update(status="passed", scratch_removed=not work.exists())
    return receipt


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--embedding":
        embedding_probe(Path(sys.argv[2]), sys.argv[3:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--launch":
        launch(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] == "readonly", sys.argv[5:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        probe(*(Path(value) for value in sys.argv[2:]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--receipt-dir", type=Path, default=Path("dist/package-evidence"))
    args = parser.parse_args()
    output = args.receipt_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = ET.Element("testsuite", name="own-package", tests="1", failures="0")
    case = ET.SubElement(suite, "testcase", name="retained-wheel-sdist")
    try:
        result = accept(Path(__file__).resolve().parents[1], args.dist_dir.resolve(), output)
    except Exception as error:
        suite.set("failures", "1")
        ET.SubElement(case, "failure").text = str(error)
        (output / "receipt.json").write_text(json.dumps({"status": "failed", "error": str(error)}))
        raise
    else:
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ET.ElementTree(suite).write(output / "junit.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
