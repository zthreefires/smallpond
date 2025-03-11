import os
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from loguru import logger

from smallpond.execution.task import (
    JobId,
    PartitionInfo,
    PerfStats,
    RuntimeContext,
    TaskId,
    TaskRuntimeId
)


def test_job_id_new():
    job_id = JobId.new()
    assert isinstance(job_id, JobId)
    assert isinstance(job_id, uuid.UUID)


def test_task_id():
    task_id = TaskId(123)
    assert str(task_id) == "000123"

    task_id = TaskId(1)
    assert str(task_id) == "000001"


def test_task_runtime_id():
    runtime_id = TaskRuntimeId(TaskId(123), 2, 1)
    assert str(runtime_id) == "000123.2.1"
    assert runtime_id.id == TaskId(123)
    assert runtime_id.epoch == 2
    assert runtime_id.retry == 1


def test_perf_stats():
    stats = PerfStats(10, 100, 5, 15, 10.0, 9.5, 12.0, 14.0, 14.8)
    assert stats.cnt == 10
    assert stats.sum == 100
    assert stats.min == 5
    assert stats.max == 15
    assert stats.avg == 10.0
    assert stats.p50 == 9.5
    assert stats.p75 == 12.0
    assert stats.p95 == 14.0
    assert stats.p99 == 14.8

    assert str(stats) == "cnt=10.0, sum=100.0, min=5.0, max=15.0, avg=10.0, p50=9.5, p75=12.0, p95=14.0, p99=14.8"


def test_partition_info():
    info = PartitionInfo(1, 4, "test_dim")
    assert info.index == 1
    assert info.npartitions == 4
    assert info.dimension == "test_dim"

    info1 = PartitionInfo(1, 4, "a")
    info2 = PartitionInfo(2, 4, "a")
    info3 = PartitionInfo(1, 4, "b")

    # Test ordering
    assert info1 < info2  # Same dimension, lower index
    assert info1 < info3  # Different dimensions, lexicographical order
    assert info2 < info3  # Different dimensions, lexicographical order


def test_runtime_context(tmp_path):
    job_id = JobId.new()
    job_time = datetime.now()
    data_root = str(tmp_path)

    ctx = RuntimeContext(job_id, job_time, data_root)

    assert ctx.job_id == job_id
    assert ctx.job_time == job_time
    assert ctx.data_root == data_root
    assert ctx.next_task_id == 0
    assert ctx.num_executors == 1
    assert isinstance(ctx.random_seed, int)

    # Test directory creation
    ctx.initialize("test_exec", root_exist_ok=True)
    assert os.path.exists(ctx.job_root)
    assert os.path.exists(ctx.config_root)
    assert os.path.exists(ctx.queue_root)
    assert os.path.exists(ctx.output_root)
    assert os.path.exists(ctx.staging_root)
    assert os.path.exists(ctx.temp_root)
    assert os.path.exists(ctx.log_root)

    # Test task ID generation
    task_id1 = ctx.new_task_id()
    task_id2 = ctx.new_task_id()
    assert task_id2 > task_id1
    assert isinstance(task_id1, TaskId)

    # Test cleanup
    ctx.cleanup()
    assert not os.path.exists(ctx.queue_root)
    assert not os.path.exists(ctx.temp_root)
    assert not os.path.exists(ctx.staging_root)
    assert not os.path.exists(ctx.output_root)


def test_runtime_context_env_overrides(tmp_path):
    job_id = JobId.new()
    job_time = datetime.now()
    data_root = str(tmp_path)

    env_overrides = {
        "TEST_VAR": "test_value",
        "LD_LIBRARY_PATH": "/test/lib"
    }

    ctx = RuntimeContext(
        job_id,
        job_time,
        data_root,
        env_overrides=env_overrides
    )

    ctx.initialize("test_exec")

    assert os.environ["TEST_VAR"] == "test_value"
    assert "/test/lib" in os.environ["LD_LIBRARY_PATH"]


def test_runtime_context_resource_limits(tmp_path):
    job_id = JobId.new()
    job_time = datetime.now()
    data_root = str(tmp_path)

    ctx = RuntimeContext(
        job_id,
        job_time,
        data_root,
        max_usable_cpu_count=2,
        max_usable_memory_size=1024*1024*1024
    )

    assert ctx.max_usable_cpu_count == 2
    assert ctx.max_usable_memory_size == 1024*1024*1024
    assert ctx.usable_cpu_count <= 2
    assert ctx.usable_memory_size <= 1024*1024*1024


def test_runtime_context_numa_binding(tmp_path):
    if os.name == "nt":
        pytest.skip("NUMA not supported on Windows")

    job_id = JobId.new()
    job_time = datetime.now()
    data_root = str(tmp_path)

    ctx = RuntimeContext(
        job_id,
        job_time,
        data_root,
        bind_numa_node=True
    )

    assert ctx.bind_numa_node == True
    assert ctx.numa_node_id is None

    # Physical resources should be divided by NUMA node count
    assert ctx.physical_cpu_count <= os.cpu_count()
    assert ctx.available_memory <= ctx.total_memory
