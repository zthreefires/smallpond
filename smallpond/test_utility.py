import io
import logging
import queue
import subprocess
import threading
import time
from unittest.mock import Mock, patch

import pytest
from loguru import logger

from smallpond.utility import (ConcurrentIter, ConcurrentIterError, InterceptHandler,
                             Wrapper, cprofile_to_string, execute_command)


def test_execute_command_success():
    lines = list(execute_command("echo hello"))
    assert lines == ["hello"]


def test_execute_command_error():
    with pytest.raises(subprocess.CalledProcessError):
        list(execute_command("ls /nonexistent"))


def test_cprofile_to_string():
    mock_profile = Mock()
    mock_stats = Mock()
    mock_profile.disable = Mock()
    mock_stats.strip_dirs.return_value = mock_stats
    mock_stats.sort_stats.return_value = mock_stats
    mock_stats.print_stats = Mock()

    with patch("pstats.Stats", return_value=mock_stats):
        result = cprofile_to_string(mock_profile)

    assert isinstance(result, str)
    mock_profile.disable.assert_called_once()
    mock_stats.strip_dirs.assert_called_once()
    mock_stats.sort_stats.assert_called_once()
    mock_stats.print_stats.assert_called_once()


def test_wrapper():
    class TestObj:
        def test_method(self):
            return "test"

        def __str__(self):
            return "TestObj"

    test_obj = TestObj()
    wrapper = Wrapper(test_obj)

    assert wrapper.test_method() == "test"
    assert str(wrapper) == "TestObj"

    wrapper.new_attr = "value"
    assert test_obj.new_attr == "value"


def test_concurrent_iter_normal():
    test_data = [1, 2, 3]
    results = []

    with ConcurrentIter(test_data) as ci:
        for item in ci:
            results.append(item)

    assert results == test_data


def test_concurrent_iter_error():
    def error_iter():
        yield 1
        raise ValueError("test error")

    with pytest.raises(ConcurrentIterError):
        with ConcurrentIter(error_iter()) as ci:
            for item in ci:
                pass


def test_concurrent_iter_early_exit():
    test_data = range(100)
    results = []

    with ConcurrentIter(test_data) as ci:
        for item in ci:
            results.append(item)
            if len(results) >= 3:
                break

    assert results == [0, 1, 2]


def test_intercept_handler():
    handler = InterceptHandler()
    record = logging.LogRecord(
        "test_logger", logging.INFO, "test.py", 10,
        "test message", (), None
    )

    with patch.object(logger, "opt") as mock_opt:
        mock_log = Mock()
        mock_opt.return_value = mock_log

        handler.emit(record)

        mock_opt.assert_called_once()
        mock_log.log.assert_called_once()


def test_intercept_handler_unknown_level():
    handler = InterceptHandler()
    record = logging.LogRecord(
        "test_logger", 123, "test.py", 10,
        "test message", (), None
    )

    with patch.object(logger, "level") as mock_level, \
         patch.object(logger, "opt") as mock_opt:
        mock_log = Mock()
        mock_opt.return_value = mock_log
        mock_level.side_effect = ValueError()

        handler.emit(record)

        mock_opt.assert_called_once()
        mock_log.log.assert_called_once()
