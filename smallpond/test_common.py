import pytest
import numpy as np
from smallpond.common import (
    clamp_value,
    clamp_row_group_size,
    clamp_row_group_bytes,
    first_value_in_dict,
    split_into_cols,
    split_into_rows,
    get_nth_partition,
    next_power_of_two,
    round_up,
    DEFAULT_ROW_GROUP_SIZE,
    DEFAULT_ROW_GROUP_BYTES,
    MAX_ROW_GROUP_SIZE,
    MAX_ROW_GROUP_BYTES,
    MB,
)

def test_clamp_value():
    assert clamp_value(5, 0, 10) == 5
    assert clamp_value(-5, 0, 10) == 0
    assert clamp_value(15, 0, 10) == 10
    assert clamp_value(5, 5, 5) == 5
    assert clamp_value(0, 0, 0) == 0
    assert clamp_value(10, 5, 15) == 10

def test_clamp_row_group_size():
    # Test values within bounds
    assert clamp_row_group_size(DEFAULT_ROW_GROUP_SIZE) == DEFAULT_ROW_GROUP_SIZE
    assert clamp_row_group_size(DEFAULT_ROW_GROUP_SIZE + 1000) == DEFAULT_ROW_GROUP_SIZE + 1000

    # Test boundary values
    assert clamp_row_group_size(MAX_ROW_GROUP_SIZE) == MAX_ROW_GROUP_SIZE
    assert clamp_row_group_size(MAX_ROW_GROUP_SIZE + 1) == MAX_ROW_GROUP_SIZE

    # Test values below minimum
    assert clamp_row_group_size(0) == DEFAULT_ROW_GROUP_SIZE
    assert clamp_row_group_size(-1) == DEFAULT_ROW_GROUP_SIZE
    assert clamp_row_group_size(DEFAULT_ROW_GROUP_SIZE - 1) == DEFAULT_ROW_GROUP_SIZE

def test_clamp_row_group_bytes():
    # Test values within bounds
    assert clamp_row_group_bytes(DEFAULT_ROW_GROUP_BYTES) == DEFAULT_ROW_GROUP_BYTES
    assert clamp_row_group_bytes(DEFAULT_ROW_GROUP_BYTES + 1000) == DEFAULT_ROW_GROUP_BYTES + 1000

    # Test boundary values
    assert clamp_row_group_bytes(MAX_ROW_GROUP_BYTES) == MAX_ROW_GROUP_BYTES
    assert clamp_row_group_bytes(MAX_ROW_GROUP_BYTES + 1) == MAX_ROW_GROUP_BYTES

    # Test values below minimum
    assert clamp_row_group_bytes(0) == DEFAULT_ROW_GROUP_BYTES
    assert clamp_row_group_bytes(-1) == DEFAULT_ROW_GROUP_BYTES
    assert clamp_row_group_bytes(DEFAULT_ROW_GROUP_BYTES - 1) == DEFAULT_ROW_GROUP_BYTES

def test_first_value_in_dict():
    # Test empty dictionary
    assert first_value_in_dict({}) is None

    # Test single item dictionary
    assert first_value_in_dict({"a": 1}) == 1
    assert first_value_in_dict({"a": None}) is None
    assert first_value_in_dict({"": ""}) == ""

    # Test multiple items dictionary
    assert first_value_in_dict({"a": 1, "b": 2}) == 1
    assert first_value_in_dict({"b": 2, "a": 1}) == 2

def test_split_into_cols():
    # Test empty list
    assert split_into_cols([], 3) == [[], [], []]

    # Test single item
    assert split_into_cols([1], 3) == [[1], [], []]

    # Test even distribution
    assert split_into_cols([1,2,3,4], 2) == [[1,3], [2,4]]

    # Test uneven distribution
    assert split_into_cols([1,2,3], 2) == [[1,3], [2]]
    assert split_into_cols([1,2], 3) == [[1], [2], []]

    # Test larger list
    assert split_into_cols(list(range(10)), 3) == [[0,3,6,9], [1,4,7], [2,5,8]]

    # Test single partition
    assert split_into_cols([1], 1) == [[1]]

def test_split_into_rows():
    # Test empty list
    assert split_into_rows([], 3) == [[], [], []]

    # Test single item
    assert split_into_rows([1], 3) == [[1], [], []]

    # Test even distribution
    assert split_into_rows([1,2,3,4], 2) == [[1,2], [3,4]]

    # Test uneven distribution
    assert split_into_rows([1,2,3], 2) == [[1,2], [3]]
    assert split_into_rows([1,2], 3) == [[1], [2], []]

    # Test larger list
    assert split_into_rows(list(range(10)), 3) == [[0,1,2,3], [4,5,6], [7,8,9]]

    # Test single partition
    assert split_into_rows([1], 1) == [[1]]

def test_get_nth_partition():
    items = list(range(10))

    # Test normal partitioning
    assert get_nth_partition(items, 0, 3) == [0,1,2,3]
    assert get_nth_partition(items, 1, 3) == [4,5,6]
    assert get_nth_partition(items, 2, 3) == [7,8,9]

    # Test edge cases
    assert get_nth_partition([], 0, 3) == []  # Empty list
    assert get_nth_partition([1], 0, 2) == [1]  # Single item
    assert get_nth_partition([1], 1, 2) == []  # Empty partition
    assert get_nth_partition([1,2,3], 0, 1) == [1,2,3]  # Single partition

    # Test invalid inputs
    with pytest.raises(ZeroDivisionError):
        get_nth_partition(items, 0, 0)  # Zero partitions

def test_next_power_of_two():
    # Test powers of 2
    assert next_power_of_two(1) == 1
    assert next_power_of_two(2) == 2
    assert next_power_of_two(4) == 4
    assert next_power_of_two(8) == 8
    assert next_power_of_two(16) == 16

    # Test non-powers of 2
    assert next_power_of_two(3) == 4
    assert next_power_of_two(5) == 8
    assert next_power_of_two(7) == 8
    assert next_power_of_two(9) == 16
    assert next_power_of_two(15) == 16

    # Test larger numbers
    assert next_power_of_two(1023) == 1024
    assert next_power_of_two(1024) == 1024
    assert next_power_of_two(1025) == 2048

def test_round_up():
    # Test with default alignment (MB)
    assert round_up(0) == 0
    assert round_up(1) == MB
    assert round_up(MB - 1) == MB
    assert round_up(MB) == MB
    assert round_up(MB + 1) == 2 * MB

    # Test with custom alignment
    assert round_up(10, 5) == 10
    assert round_up(11, 5) == 15
    assert round_up(0, 5) == 0
    assert round_up(4, 5) == 5
    assert round_up(-1, 5) == 0

    # Test larger alignments
    assert round_up(100, 32) == 128
    assert round_up(1000, 256) == 1024
