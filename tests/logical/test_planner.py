import pytest
from unittest.mock import Mock, MagicMock, patch
import os
from smallpond.execution.task import *
from smallpond.logical.node import *
from smallpond.logical.planner import Planner

@pytest.fixture
def runtime_ctx():
    ctx = Mock()
    ctx.final_output_path = None
    ctx.output_root = "/tmp/output"
    ctx.num_executors = 2
    ctx.usable_cpu_count = 4
    return ctx

@pytest.fixture
def planner(runtime_ctx):
    return Planner(runtime_ctx)

@pytest.fixture
def mock_logical_context():
    return Mock()

@pytest.fixture
def mock_task():
    task = Mock()
    task.partition_dims = ('dim1',)
    task.partition_infos = [PartitionInfo()]
    return task

def test_broadcast_input_deps_no_inputs(planner):
    node = Mock()
    node.input_deps = []

    results = list(planner.broadcast_input_deps(node, 0))
    assert len(results) == 1
    assert results[0] == ([], (PartitionInfo(),))
