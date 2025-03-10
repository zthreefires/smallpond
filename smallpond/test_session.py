import os
from datetime import datetime
import pytest
from unittest.mock import MagicMock, patch
from smallpond.session import Config
from smallpond.platform import Platform

@pytest.fixture
def mock_platform():
    platform = MagicMock()
    platform.__str__ = lambda x: "mock_platform"
    platform.default_job_id.return_value = "test_job"
    platform.default_job_time.return_value = datetime(2025, 3, 1)
    platform.default_data_root.return_value = "/data"
    platform.default_memory_allocator.return_value = "system"
    return platform

@pytest.fixture
def mock_get_platform():
    with patch('smallpond.session.get_platform') as mock:
        def _get_platform(platform_str):
            if isinstance(platform_str, Platform):
                return platform_str
            mock_platform = MagicMock()
            mock_platform.__str__ = lambda x: str(platform_str)
            mock_platform.default_job_id.return_value = "test_job"
            mock_platform.default_job_time.return_value = datetime(2025, 3, 1)
            mock_platform.default_data_root.return_value = "/data"
            mock_platform.default_memory_allocator.return_value = "system"
            return mock_platform
        mock.side_effect = _get_platform
        yield mock

def test_config_from_args_and_env_defaults(mock_platform, mock_get_platform):
    config, platform = Config.from_args_and_env(platform=mock_platform)

    assert config.job_id == "test_job"
    assert config.job_time == datetime(2025, 3, 1)
    assert config.data_root == "/data"
    assert config.num_executors == 0
    assert config.ray_address is None
    assert not config.bind_numa_node
    assert config.memory_allocator == "system"
    assert config.remove_output_root is True

def test_config_from_args_and_env_args(mock_platform, mock_get_platform):
    config, platform = Config.from_args_and_env(
        platform=mock_platform,
        job_id="custom_job",
        job_time=datetime(2025, 3, 2),
        data_root="/custom/data",
        num_executors=4,
        ray_address="localhost:6379",
        bind_numa_node=True,
        memory_allocator="jemalloc",
        _remove_output_root=False
    )

    assert config.job_id == "custom_job"
    assert config.job_time == datetime(2025, 3, 2)
    assert config.data_root == "/custom/data"
    assert config.num_executors == 4
    assert config.ray_address == "localhost:6379"
    assert config.bind_numa_node is True
    assert config.memory_allocator == "jemalloc"
    assert config.remove_output_root is False

def test_config_from_args_and_env_env_vars(mock_platform, mock_get_platform, monkeypatch):
    monkeypatch.setenv("SP_JOBID", "env_job")
    monkeypatch.setenv("SP_JOB_TIME", "2025-03-03T12:00:00")
    monkeypatch.setenv("SP_DATA_ROOT", "/env/data")
    monkeypatch.setenv("SP_NUM_EXECUTORS", "8")
    monkeypatch.setenv("SP_RAY_ADDRESS", "ray://cluster")
    monkeypatch.setenv("SP_BIND_NUMA_NODE", "1")
    monkeypatch.setenv("SP_MEMORY_ALLOCATOR", "mimalloc")

    config, platform = Config.from_args_and_env(platform=mock_platform)

    assert config.job_id == "env_job"
    assert config.job_time == datetime(2025, 3, 3, 12)
    assert config.data_root == "/env/data"
    assert config.num_executors == 8
    assert config.ray_address == "ray://cluster"
    assert config.bind_numa_node is True
    assert config.memory_allocator == "mimalloc"

def test_config_from_args_and_env_platform_str(mock_get_platform):
    config, platform = Config.from_args_and_env(platform="test_platform")

    assert str(platform) == "test_platform"
    assert config.job_id == "test_job"
