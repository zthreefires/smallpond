import pytest
from unittest.mock import Mock, patch
import copy
from smallpond.logical.optimizer import Optimizer
from smallpond.logical.node import Node, SqlEngineNode
from typing import Set


@pytest.fixture
def mock_node():
    node = Mock(spec=Node)
    node.input_deps = []
    return node


@pytest.fixture
def optimizer():
    exclude_nodes: Set[Node] = set()
    return Optimizer(exclude_nodes)


@pytest.fixture
def mock_sql_engine_node():
    node = Mock(spec=SqlEngineNode)
    node.input_deps = []
    node.udfs = []
    node.cpu_limit = 1
    node.gpu_limit = 0
    node.memory_limit = 1000
    node.sql_queries = ["SELECT * FROM table"]
    return node


def test_visit_excluded_node(optimizer, mock_node):
    optimizer.exclude_nodes.add(mock_node)
    result = optimizer.visit(mock_node)
    assert result == mock_node


def test_visit_memoized_node(optimizer, mock_node):
    optimizer.optimized_node_map[mock_node] = mock_node
    result = optimizer.visit(mock_node)
    assert result == mock_node


def test_generic_visit_empty_deps(optimizer, mock_node):
    result = optimizer.generic_visit(mock_node, 0)
    assert isinstance(result, Node)
    assert len(result.input_deps) == 0


def test_visit_query_engine_node_fusion(optimizer):
    child = Mock(spec=SqlEngineNode)
    child.input_deps = []
    child.sql_queries = ["SELECT * FROM {0}"]
    child.udfs = ["udf1"]
    child.cpu_limit = 1
    child.gpu_limit = 0
    child.memory_limit = 1000

    parent = Mock(spec=SqlEngineNode)
    parent.input_deps = [child]
    parent.udfs = ["udf2"]
    parent.cpu_limit = 2
    parent.gpu_limit = 1
    parent.memory_limit = 2000
    parent.sql_queries = ["SELECT a, b FROM {0}"]

    result = optimizer.visit_query_engine_node(parent, 0)

    assert isinstance(result, SqlEngineNode)
    assert result.udfs == ["udf2", "udf1"]
    assert result.cpu_limit == 2
    assert result.gpu_limit == 1
    assert result.memory_limit == 2000
    assert result.sql_queries == ["SELECT a, b FROM (SELECT * FROM {0})"]


def test_visit_query_engine_node_memory_limit_none(optimizer):
    child = Mock(spec=SqlEngineNode)
    child.input_deps = []
    child.memory_limit = None
    child.sql_queries = ["SELECT * FROM {0}"]
    child.udfs = []
    child.cpu_limit = 1
    child.gpu_limit = 0

    parent = Mock(spec=SqlEngineNode)
    parent.input_deps = [child]
    parent.memory_limit = 1000
    parent.sql_queries = ["SELECT a FROM {0}"]
    parent.udfs = []
    parent.cpu_limit = 1
    parent.gpu_limit = 0

    result = optimizer.visit_query_engine_node(parent, 0)
    assert result.memory_limit == 1000


def test_visit_query_engine_node_multiple_queries(optimizer):
    child = Mock(spec=SqlEngineNode)
    child.input_deps = []
    child.sql_queries = ["SELECT * FROM t1", "SELECT * FROM {0}"]
    child.udfs = []
    child.cpu_limit = 1
    child.gpu_limit = 0
    child.memory_limit = 1000

    parent = Mock(spec=SqlEngineNode)
    parent.input_deps = [child]
    parent.sql_queries = ["SELECT a FROM {0}"]
    parent.udfs = []
    parent.cpu_limit = 1
    parent.gpu_limit = 0
    parent.memory_limit = 1000

    result = optimizer.visit_query_engine_node(parent, 0)
    assert result.sql_queries == ["SELECT * FROM t1", "SELECT a FROM (SELECT * FROM {0})"]


def test_visit_query_engine_node_no_fusion_multiple_inputs(optimizer, mock_sql_engine_node):
    node = Mock(spec=SqlEngineNode)
    node.input_deps = [mock_sql_engine_node, mock_sql_engine_node]
    node.sql_queries = ["SELECT * FROM {0}"]
    node.udfs = []
    node.cpu_limit = 1
    node.gpu_limit = 0
    node.memory_limit = 1000

    result = optimizer.visit_query_engine_node(node, 0)
    assert len(result.input_deps) == 2
    assert result.sql_queries == ["SELECT * FROM {0}"]


def test_visit_with_deep_dependency_chain(optimizer):
    leaf = Mock(spec=SqlEngineNode)
    leaf.input_deps = []
    leaf.sql_queries = ["SELECT * FROM table"]
    leaf.udfs = []
    leaf.cpu_limit = 1
    leaf.gpu_limit = 0
    leaf.memory_limit = 1000

    middle = Mock(spec=SqlEngineNode)
    middle.input_deps = [leaf]
    middle.sql_queries = ["SELECT a FROM {0}"]
    middle.udfs = []
    middle.cpu_limit = 1
    middle.gpu_limit = 0
    middle.memory_limit = 1000

    root = Mock(spec=SqlEngineNode)
    root.input_deps = [middle]
    root.sql_queries = ["SELECT b FROM {0}"]
    root.udfs = []
    root.cpu_limit = 1
    root.gpu_limit = 0
    root.memory_limit = 1000

    result = optimizer.visit(root)
    assert isinstance(result, SqlEngineNode)
    assert result.sql_queries == ["SELECT b FROM (SELECT a FROM (SELECT * FROM table))"]


def test_visit_query_engine_node_both_memory_limits_none(optimizer):
    child = Mock(spec=SqlEngineNode)
    child.input_deps = []
    child.memory_limit = None
    child.sql_queries = ["SELECT * FROM {0}"]
    child.udfs = []
    child.cpu_limit = 1
    child.gpu_limit = 0

    parent = Mock(spec=SqlEngineNode)
    parent.input_deps = [child]
    parent.memory_limit = None
    parent.sql_queries = ["SELECT a FROM {0}"]
    parent.udfs = []
    parent.cpu_limit = 1
    parent.gpu_limit = 0

    result = optimizer.visit_query_engine_node(parent, 0)
    assert result.memory_limit is None
