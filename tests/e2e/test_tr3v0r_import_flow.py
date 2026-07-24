from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from local_rag_backend.cli import cli
from local_rag_backend.core.domain.retrieval import RetrievalFilter, RetrievalRequest
from local_rag_backend.infrastructure.persistence.sql import SqlDocumentStorage
from local_rag_backend.infrastructure.search_backends.local_split import LocalSplitSearchRetriever
from local_rag_backend.settings import settings

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
TR3V0R_SRC = WORKSPACE_ROOT / "tr3v0r" / "src"
pytest.importorskip("pyarrow")
if str(TR3V0R_SRC) not in sys.path:
    sys.path.insert(0, str(TR3V0R_SRC))


def _load_tr3v0r_symbols() -> dict[str, object]:
    graph_module = importlib.import_module("tr3v0r.builders.graph")
    canonical_rag_module = importlib.import_module("tr3v0r.canonical_rag")
    contracts_module = importlib.import_module("tr3v0r.contracts")
    io_module = importlib.import_module("tr3v0r.io")
    return {
        "GroupByMac": graph_module.GroupByMac,
        "build_net_nodes_with_index_stream": graph_module.build_net_nodes_with_index_stream,
        "build_canonical_rag_payload": canonical_rag_module.build_canonical_rag_payload,
        "FeatureWindowV1": contracts_module.FeatureWindowV1,
        "GraphSnapshotV1": contracts_module.GraphSnapshotV1,
        "NetEdgeV1": contracts_module.NetEdgeV1,
        "write_feature_windows": io_module.write_feature_windows,
        "write_graph_snapshots": io_module.write_graph_snapshots,
        "write_net_edges": io_module.write_net_edges,
        "write_net_nodes": io_module.write_net_nodes,
    }


def _feature_window(
    *,
    sensor_id: str,
    window_start_ns: int,
    window_end_ns: int,
) -> object:
    FeatureWindowV1 = _load_tr3v0r_symbols()["FeatureWindowV1"]
    return FeatureWindowV1(
        sensor_id=sensor_id,
        entity_id="aa:bb:cc:dd:ee:ff",
        phy="wifi",
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        window_id=f"{sensor_id}:aa:bb:cc:dd:ee:ff:{window_start_ns}",
        frame_count=12,
        unique_peer_count=3,
        mgmt_frame_count=4,
        data_frame_count=8,
        mean_rssi_dbm=-48.5,
        std_rssi_dbm=1.5,
        mean_interarrival_ns=5000.0,
        burstiness=0.42,
        mgmt_to_data_ratio=0.5,
        ie_entropy=1.25,
        adv_type_entropy=None,
        seen_on_bands=1,
        ssid_fingerprint=101,
        capability_fingerprint=202,
        behavior_fingerprint=303,
    )


def _write_tr3v0r_payload(payload_path: Path, built_dir: Path) -> dict[str, object]:
    symbols = _load_tr3v0r_symbols()
    GroupByMac = symbols["GroupByMac"]
    build_net_nodes_with_index_stream = symbols["build_net_nodes_with_index_stream"]
    NetEdgeV1 = symbols["NetEdgeV1"]
    GraphSnapshotV1 = symbols["GraphSnapshotV1"]
    write_feature_windows = symbols["write_feature_windows"]
    write_net_nodes = symbols["write_net_nodes"]
    write_net_edges = symbols["write_net_edges"]
    write_graph_snapshots = symbols["write_graph_snapshots"]
    build_canonical_rag_payload = symbols["build_canonical_rag_payload"]
    feature_windows = [
        _feature_window(
            sensor_id="sensor_a",
            window_start_ns=1_000_000_000,
            window_end_ns=2_000_000_000,
        ),
        _feature_window(
            sensor_id="sensor_b",
            window_start_ns=2_000_000_000,
            window_end_ns=3_000_000_000,
        ),
    ]
    nodes, _entity_index = build_net_nodes_with_index_stream(
        feature_windows,
        strategy=GroupByMac(),
    )
    write_feature_windows(feature_windows, built_dir / "feature_windows.parquet")
    write_net_nodes(nodes, built_dir / "net_nodes.parquet")
    write_net_edges(
        [
            NetEdgeV1(
                edge_id="edge-1",
                sensor_id="sensor_a",
                src_node_id="node-client-1",
                dst_node_id="aa:bb:cc:dd:ee:ff",
                edge_type="wifi_data_flow",
                window_start_ns=1_000_000_000,
                window_end_ns=2_000_000_000,
                window_id="node-client-1:aa:bb:cc:dd:ee:ff:1000000000",
                frame_count=9,
                mean_rssi_dbm=-49.0,
                std_rssi_dbm=1.1,
                mean_interarrival_ns=6000.0,
                burstiness=0.12,
                security_mismatch_flag=False,
                anomalous_behavior_flag=False,
            )
        ],
        built_dir / "net_edges.parquet",
    )
    write_graph_snapshots(
        [
            GraphSnapshotV1(
                snapshot_id="sensor_a:1000000000:2000000000",
                sensor_id="sensor_a",
                window_start_ns=1_000_000_000,
                window_end_ns=2_000_000_000,
                node_count=2,
                edge_count=1,
                mean_degree=1.0,
                max_degree=1,
                component_count=1,
            )
        ],
        built_dir / "graph_snapshots.parquet",
    )
    payload = build_canonical_rag_payload(
        input_dir=built_dir,
        dataset_key="RF Lab Demo",
        identity_mode="mac_strict",
    )
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def test_tr3v0r_canonical_import_preserves_net_node_sensor_ids_and_retrieves_by_membership(
    in_memory_sqlite, tmp_path, monkeypatch
):
    _ = in_memory_sqlite
    monkeypatch.setattr(settings, "retrieval_mode", "sparse", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "data_dir", data_dir, raising=False)

    built_dir = tmp_path / "built"
    built_dir.mkdir()
    payload_path = tmp_path / "tr3v0r_canonical.json"
    payload = _write_tr3v0r_payload(payload_path, built_dir)

    assert payload["scope"] == "tr3v0r:rf-lab-demo"
    assert payload["replace_scope"] is True

    result = CliRunner().invoke(cli, ["import-canonical", "--json", str(payload_path)])
    assert result.exit_code == 0, result.output

    repo = SqlDocumentStorage()
    docs = list(repo.get_all_documents())
    assert len(docs) == 6

    retriever = LocalSplitSearchRetriever(doc_repo=repo)
    retrieval = retriever.retrieve(
        RetrievalRequest(
            query="wifi_ap sensor_a sensor_b",
            top_k=1,
            mode="sparse",
            filters=(
                RetrievalFilter(field="metadata.doc_type", values=("net_node",)),
                RetrievalFilter(field="metadata.sensor_ids", values=("sensor_a",)),
            ),
        )
    )

    assert retrieval.items
    document = retrieval.items[0].document
    metadata = dict(document.metadata or {})
    assert document.external_id == "tr3v0r:rf-lab-demo:node:aa:bb:cc:dd:ee:ff"
    assert metadata["doc_type"] == "net_node"
    assert metadata["sensor_ids"] == ["sensor_a", "sensor_b"]
