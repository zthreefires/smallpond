import os
import pytest
import pandas as pd
import pyarrow as pa
import duckdb
from unittest.mock import patch, mock_open, MagicMock
from collections import OrderedDict

from smallpond.logical.dataset import (
    DataSet,
    CsvDataSet,
    JsonDataSet,
    ParquetDataSet,
    SqlQueryDataSet,
    ArrowTableDataSet,
    PandasDataSet,
    PartitionedDataSet,
    FileSet
)

@pytest.fixture
def sample_dataset():
    return DataSet(['test.parquet'], root_dir='/data', recursive=False)

@pytest.fixture
def tmp_parquet_file(tmp_path):
    table = pa.table({'col1': [1,2,3], 'col2': ['a','b','c']})
    path = tmp_path / 'test.parquet'
    pa.parquet.write_table(table, str(path))
    return str(path)

@pytest.fixture
def mock_duckdb_conn():
    conn = MagicMock()
    conn.sql.return_value.fetch_arrow_table.return_value = pa.table({'col1': [1,2,3]})
    schema = pa.schema([('col1', pa.int64())])
    batch = pa.RecordBatch.from_arrays([pa.array([1,2,3])], names=['col1'])
    conn.sql.return_value.fetch_arrow_reader.return_value = pa.RecordBatchReader.from_batches(schema, [batch])
    return conn

def test_dataset_reset(sample_dataset):
    sample_dataset.reset(['new.parquet'], '/newdata', True)
    assert sample_dataset.paths == ['new.parquet']
    assert sample_dataset.root_dir == os.path.abspath('/newdata')
    assert sample_dataset.recursive == True
    assert sample_dataset._resolved_paths is None
    assert sample_dataset._absolute_paths is None

def test_dataset_partition_by_files():
    ds = DataSet(['file1.parquet', 'file2.parquet', 'file3.parquet'])
    partitions = ds.partition_by_files(2)
    assert len(partitions) == 2
    assert len(partitions[0].paths) <= 2
    assert len(partitions[1].paths) <= 2

def test_dataset_partition_by_files_random():
    ds = DataSet(['file1.parquet', 'file2.parquet', 'file3.parquet'])
    partitions = ds.partition_by_files(2, random_shuffle=True)
    assert len(partitions) == 2
    total_files = sum(len(p.paths) for p in partitions)
    assert total_files == 3

def test_csv_dataset_sql_query():
    schema = OrderedDict([('col1', 'INTEGER'), ('col2', 'VARCHAR')])
    ds = CsvDataSet(['data.csv'], schema, delim=',', max_line_size=1024, parallel=True, header=True)
    query = ds.sql_query_fragment()
    assert 'read_csv' in query
    assert "delim=','" in query
    assert "'col1': 'INTEGER'" in query
    assert "'col2': 'VARCHAR'" in query
    assert 'max_line_size=1024' in query
    assert 'parallel=True' in query
    assert 'header=True' in query

def test_json_dataset_sql_query():
    schema = OrderedDict([('col1', 'INTEGER'), ('col2', 'VARCHAR')])
    ds = JsonDataSet(['data.json'], schema, format='newline_delimited', max_object_size=1024)
    query = ds.sql_query_fragment()
    assert 'read_json' in query
    assert "'col1': 'INTEGER'" in query
    assert "'col2': 'VARCHAR'" in query
    assert "format='newline_delimited'" in query
    assert 'maximum_object_size=1024' in query

def test_parquet_dataset_reset():
    ds = ParquetDataSet(['data.parquet'], columns=['col1'], generated_columns=['file_name'])
    ds.reset(['new.parquet'], '/newdata', True)
    assert ds.paths == ['new.parquet']
    assert ds._resolved_row_ranges is None
    assert ds._resolved_paths is None
    assert ds._absolute_paths is None
    assert ds.columns == ['col1']
    assert ds.generated_columns == ['file_name']

def test_parquet_dataset_sql_query(tmp_parquet_file):
    ds = ParquetDataSet([tmp_parquet_file], columns=['col1'], generated_columns=['file_name'])
    query = ds.sql_query_fragment()
    assert 'read_parquet' in query
    assert tmp_parquet_file in query
    assert 'file_name=true' in query

def test_parquet_dataset_partition_by_rows(tmp_parquet_file):
    ds = ParquetDataSet([tmp_parquet_file])
    partitions = ds.partition_by_rows(2)
    assert len(partitions) == 2
    assert all(isinstance(p, ParquetDataSet) for p in partitions)
    assert all(len(p.resolved_paths) > 0 for p in partitions)

def test_parquet_dataset_partition_by_size(tmp_parquet_file):
    ds = ParquetDataSet([tmp_parquet_file])
    partitions = ds.partition_by_size(1024*1024) # 1MB
    assert len(partitions) > 0
    assert all(isinstance(p, ParquetDataSet) for p in partitions)
    assert all(len(p.resolved_paths) > 0 for p in partitions)

def test_sql_query_dataset_sql_query():
    ds = SqlQueryDataSet("SELECT * FROM table")
    query = ds.sql_query_fragment()
    assert "SELECT * FROM table" in query

def test_sql_query_dataset_with_builder():
    def query_builder(conn, fs):
        return "SELECT * FROM custom_table"
    ds = SqlQueryDataSet("SELECT 1", query_builder)
    query = ds.sql_query_fragment(None, None)
    assert "SELECT * FROM custom_table" in query

def test_arrow_table_dataset_methods(mock_duckdb_conn):
    table = pa.table({'col1': [1,2,3]})
    ds = ArrowTableDataSet(table)

    assert isinstance(ds.to_arrow_table(), pa.Table)
    assert isinstance(ds.to_batch_reader(), pa.RecordBatchReader)

    query = ds.sql_query_fragment(conn=mock_duckdb_conn)
    assert 'arrow_table_' in query

def test_pandas_dataset_methods(mock_duckdb_conn):
    df = pd.DataFrame({'col1': [1,2,3]})
    ds = PandasDataSet(df)

    assert isinstance(ds.to_pandas(), pd.DataFrame)
    assert isinstance(ds.to_arrow_table(), pa.Table)
    assert isinstance(ds.to_batch_reader(), pa.RecordBatchReader)

    query = ds.sql_query_fragment(conn=mock_duckdb_conn)
    assert 'pandas_table_' in query

def test_fileset_methods():
    fs = FileSet(['file1.txt', 'file2.txt'])
    table = fs.to_arrow_table()
    assert isinstance(table, pa.Table)
    assert len(table) == 2
    assert table.column_names == ['resolved_paths']

    reader = fs.to_batch_reader()
    assert isinstance(reader, pa.RecordBatchReader)

def test_partitioned_dataset():
    ds1 = ParquetDataSet(['file1.parquet'])
    ds2 = ParquetDataSet(['file2.parquet'])
    pds = PartitionedDataSet([ds1, ds2])

    assert len(pds.datasets) == 2
    assert pds[0] == ds1
    assert pds[1] == ds2
    assert len(pds.paths) == 2
    assert pds.columns == ds1.columns
