import os
import sys
import socket
from unittest.mock import Mock, patch

import pytest

from smallpond.execution.driver import Driver
from smallpond.logical.node import LogicalPlan


@pytest.fixture
def driver():
    return Driver()


def test_add_argument(driver):
    driver.add_argument("--test", help="test argument")
    driver.add_argument("--test2", type=int, default=42)

    args = driver.user_args_parser.parse_args(["--test", "value"])
    assert args.test == "value"
    assert args.test2 == 42


def test_parse_arguments_with_args(driver):
    test_args = [
        "scheduler",
        "--job_id", "test_job",
        "--job_name", "test_name",
        "--num_executors", "5",
        "--max_retry", "3",
        "--enable_profiling"
    ]
    user_args, driver_args = driver.parse_arguments(test_args)

    assert driver_args.mode == "scheduler"
    assert driver_args.job_id == "test_job"
    assert driver_args.job_name == "test_name"
    assert driver_args.num_executors == 5
    assert driver_args.max_retry_count == 3
    assert driver_args.enable_profiling is True


def test_get_user_arguments(driver):
    driver.add_argument("--user_arg", default="test")
    driver.add_argument("--user_flag", action="store_true")

    driver.parse_arguments(["executor"])

    args = driver.get_user_arguments()
    assert isinstance(args, dict)
    assert args["user_arg"] == "test"
    assert args["user_flag"] is False

    args = driver.get_user_arguments(to_dict=False)
    assert not isinstance(args, dict)
    assert args.user_arg == "test"
    assert args.user_flag is False


def test_get_driver_arguments(driver):
    test_args = ["executor", "--job_name", "test_job"]
    driver.parse_arguments(test_args)

    args = driver.get_driver_arguments()
    assert isinstance(args, dict)
    assert args["mode"] == "executor"
    assert args["job_name"] == "test_job"

    args = driver.get_driver_arguments(to_dict=False)
    assert not isinstance(args, dict)
    assert args.mode == "executor"
    assert args.job_name == "test_job"


def test_run_missing_ctx_file(driver):
    with pytest.raises(AssertionError):
        driver.parse_arguments(["executor", "--runtime_ctx_path", "/nonexistent/path"])
        driver.run(None)


def test_run_scheduler_without_plan(driver):
    with pytest.raises(AssertionError):
        driver.parse_arguments(["scheduler"])
        driver.run(None)
