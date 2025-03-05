import shutil
import pytest
from unittest.mock import patch
from smallpond.platform.mpi import MPI

def test_is_available_when_mpirun_exists():
    with patch('shutil.which', return_value='/usr/bin/mpirun'):
        assert MPI.is_available() is True

def test_is_available_when_mpirun_not_exists():
    with patch('shutil.which', return_value=None):
        assert MPI.is_available() is False

def test_is_available_calls_shutil_which():
    with patch('shutil.which') as mock_which:
        MPI.is_available()
        mock_which.assert_called_once_with('mpirun')
