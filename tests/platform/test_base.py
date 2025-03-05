import os
import signal
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from smallpond.platform.base import Platform


@pytest.fixture
def platform():
    return Platform()


def test_start_job_basic(platform):
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pids = platform.start_job(
            num_nodes=1,
            entrypoint="test.py",
            args=["--arg1", "val1"],
            envs={"ENV1": "val1"},
            extra_opts={}
        )

        assert len(pids) == 1
        assert pids[0] == "12345"
        mock_popen.assert_called_once_with(
            ["python", "test.py", "--arg1", "val1"],
            env={**os.environ, "ENV1": "val1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )


def test_start_job_multiple_nodes(platform):
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pids = platform.start_job(
            num_nodes=3,
            entrypoint="test.py",
            args=[],
            envs={},
            extra_opts={}
        )

        assert len(pids) == 3
        assert all(pid == "12345" for pid in pids)
        assert mock_popen.call_count == 3


def test_stop_job(platform):
    with patch('os.kill') as mock_kill:
        platform.stop_job("12345")
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)


def test_stop_job_invalid_pid(platform):
    with pytest.raises(ValueError):
        platform.stop_job("invalid_pid")


def test_start_job_empty_args(platform):
    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pids = platform.start_job(
            num_nodes=1,
            entrypoint="test.py",
            args=[],
            envs={},
            extra_opts={}
        )

        assert len(pids) == 1
        mock_popen.assert_called_once_with(
            ["python", "test.py"],
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )


def test_start_job_with_env_override(platform, monkeypatch):
    monkeypatch.setenv('EXISTING_ENV', 'original')

    with patch('subprocess.Popen') as mock_popen:
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pids = platform.start_job(
            num_nodes=1,
            entrypoint="test.py",
            args=[],
            envs={'EXISTING_ENV': 'override'},
            extra_opts={}
        )

        assert len(pids) == 1
        called_env = mock_popen.call_args[1]['env']
        assert called_env['EXISTING_ENV'] == 'override'
