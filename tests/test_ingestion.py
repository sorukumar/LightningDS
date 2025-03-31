"""
Tests for the ingestion module.
"""
import os
import json
import tempfile
import pandas as pd
import pytest
from lightningDS import ingestion


def test_load_json_data():
    """Test loading JSON data from a file."""
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_data = {
            "nodes": [{"pub_key": "node1", "alias": "test1"}, {"pub_key": "node2", "alias": "test2"}],
            "edges": [{"node1_pub": "node1", "node2_pub": "node2", "capacity": 1000}]
        }
        f.write(json.dumps(test_data).encode('utf-8'))
        temp_file = f.name

    try:
        # Test loading the file
        loaded_data = ingestion.load_json_data(temp_file)
        assert loaded_data == test_data
        assert len(loaded_data["nodes"]) == 2
        assert len(loaded_data["edges"]) == 1
    finally:
        # Clean up
        os.unlink(temp_file)


def test_load_json_data_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        ingestion.load_json_data("nonexistent_file.json")


def test_json_to_csv():
    """Test converting JSON data to CSV files."""
    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_data = {
            "nodes": [{"pub_key": "node1", "alias": "test1"}, {"pub_key": "node2", "alias": "test2"}],
            "edges": [{"node1_pub": "node1", "node2_pub": "node2", "capacity": 1000}]
        }
        f.write(json.dumps(test_data).encode('utf-8'))
        temp_file = f.name

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test converting to CSV
            nodes_csv, channels_csv = ingestion.json_to_csv(temp_file, temp_dir, prefix="test_")

            # Check that files were created
            assert os.path.exists(nodes_csv)
            assert os.path.exists(channels_csv)

            # Check file contents
            nodes_df = pd.read_csv(nodes_csv)
            channels_df = pd.read_csv(channels_csv)

            assert len(nodes_df) == 2
            assert len(channels_df) == 1
            assert "pub_key" in nodes_df.columns
            assert "node1_pub" in channels_df.columns
        finally:
            # Clean up
            os.unlink(temp_file)


def test_load_csv_data():
    """Test loading data from a CSV file."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        test_data = pd.DataFrame({
            "pub_key": ["node1", "node2"],
            "alias": ["test1", "test2"]
        })
        test_data.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        # Test loading the file
        loaded_data = ingestion.load_csv_data(temp_file)
        assert len(loaded_data) == 2
        assert "pub_key" in loaded_data.columns
        assert loaded_data["pub_key"].tolist() == ["node1", "node2"]
    finally:
        # Clean up
        os.unlink(temp_file)


def test_load_csv_data_file_not_found():
    """Test handling of non-existent CSV files."""
    with pytest.raises(FileNotFoundError):
        ingestion.load_csv_data("nonexistent_file.csv")