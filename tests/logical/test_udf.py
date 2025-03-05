import pytest
import duckdb
from unittest.mock import Mock, patch
from smallpond.logical.udf import (
    UDFType,
    UDFStructType,
    UDFListType,
    UDFMapType,
    UDFAnyParameters,
    PythonUDFContext,
    ExternalModuleContext,
    DuckDbExtensionContext,
    UserDefinedFunction,
    udf
)

def test_udf_type_to_duckdb():
    assert UDFType.VARCHAR.to_duckdb_type() == duckdb.typing.VARCHAR
    assert UDFType.INTEGER.to_duckdb_type() == duckdb.typing.INTEGER
    assert UDFType.DOUBLE.to_duckdb_type() == duckdb.typing.DOUBLE

def test_udf_struct_type():
    fields = {'name': 'VARCHAR', 'age': 'INTEGER'}
    struct_type = UDFStructType(fields)
    with patch('duckdb.struct_type') as mock_struct:
        struct_type.to_duckdb_type()
        mock_struct.assert_called_once_with(fields)

def test_udf_list_type():
    child_type = UDFType.INTEGER
    list_type = UDFListType(child_type)
    with patch('duckdb.list_type') as mock_list:
        list_type.to_duckdb_type()
        mock_list.assert_called_once_with(duckdb.typing.INTEGER)

def test_udf_map_type():
    key_type = UDFType.VARCHAR
    value_type = UDFType.INTEGER
    map_type = UDFMapType(key_type, value_type)
    with patch('duckdb.map_type') as mock_map:
        map_type.to_duckdb_type()
        mock_map.assert_called_once_with(duckdb.typing.VARCHAR, duckdb.typing.INTEGER)

def test_udf_any_parameters():
    any_params = UDFAnyParameters()
    assert any_params.to_duckdb_type() is None

def test_python_udf_context_bind():
    def test_func(x): return x + 1

    context = PythonUDFContext(
        "test_func",
        test_func,
        [UDFType.INTEGER],
        UDFType.INTEGER
    )

    mock_conn = Mock()
    context.bind(mock_conn)

    mock_conn.create_function.assert_called_once_with(
        "test_func",
        test_func,
        [duckdb.typing.INTEGER],
        duckdb.typing.INTEGER,
        type="native"
    )

def test_python_udf_context_bind_arrow():
    def test_func(x): return x + 1

    context = PythonUDFContext(
        "test_func",
        test_func,
        [UDFType.INTEGER],
        UDFType.INTEGER,
        use_arrow_type=True
    )

    mock_conn = Mock()
    context.bind(mock_conn)

    mock_conn.create_function.assert_called_once_with(
        "test_func",
        test_func,
        [duckdb.typing.INTEGER],
        duckdb.typing.INTEGER,
        type="arrow"
    )

def test_python_udf_context_bind_any_params():
    def test_func(x): return x + 1

    context = PythonUDFContext(
        "test_func",
        test_func,
        UDFAnyParameters(),
        UDFType.INTEGER
    )

    mock_conn = Mock()
    context.bind(mock_conn)

    mock_conn.create_function.assert_called_once_with(
        "test_func",
        test_func,
        None,
        duckdb.typing.INTEGER,
        type="native"
    )

def test_external_module_context_bind(tmp_path):
    module_content = """
def create_duckdb_udfs(conn):
    pass
udfs = []
"""
    module_path = tmp_path / "test_module.py"
    module_path.write_text(module_content)

    context = ExternalModuleContext("test_module", str(module_path))
    mock_conn = Mock()

    context.bind(mock_conn)

def test_duckdb_extension_context_bind():
    context = DuckDbExtensionContext("test_ext", "test_path")
    mock_conn = Mock()

    context.bind(mock_conn)
    mock_conn.load_extension.assert_called_once_with("test_path")

def test_udf_decorator():
    @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER)
    def test_func(x):
        return x + 1

    assert isinstance(test_func, UserDefinedFunction)
    assert test_func.name == "test_func"
    assert test_func.params == [UDFType.INTEGER]
    assert test_func.return_type == UDFType.INTEGER
    assert test_func.use_arrow_type == False

def test_udf_decorator_with_name():
    @udf(params=[UDFType.INTEGER], return_type=UDFType.INTEGER, name="custom_name")
    def test_func(x):
        return x + 1

    assert test_func.name == "custom_name"
