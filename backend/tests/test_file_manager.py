"""Tests for file manager utilities"""
import pytest
from pathlib import Path
import tempfile
import shutil
from backend.utils.file_manager import (
    sanitize_filename,
    validate_path,
    ensure_directories,
    get_file_size_mb
)


class TestFileManager:
    """Test file manager utilities"""
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Valid filename
        assert sanitize_filename("test_scenario") == "test_scenario"
        
        # Filename with spaces
        assert sanitize_filename("test scenario") == "test_scenario"
        
        # Filename with special characters
        assert sanitize_filename("test@scenario#123") == "test_scenario_123"
        
        # Filename with path traversal attempt
        assert sanitize_filename("../../../etc/passwd") == "etc_passwd"
    
    def test_validate_path(self):
        """Test path validation"""
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Valid path within allowed directory
            valid_path = temp_dir / "test.yaml"
            assert validate_path(valid_path, temp_dir) is True
            
            # Invalid path outside allowed directory
            invalid_path = Path("/etc/passwd")
            assert validate_path(invalid_path, temp_dir) is False
            
            # Path traversal attempt
            traversal_path = temp_dir / ".." / ".." / "etc" / "passwd"
            assert validate_path(traversal_path, temp_dir) is False
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_ensure_directories(self):
        """Test directory creation"""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test creating nested directories
            test_path = temp_dir / "configs" / "scenarios"
            ensure_directories()
            
            # Verify standard directories exist
            assert Path("configs").exists()
            assert Path("models").exists()
            assert Path("results").exists()
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_get_file_size_mb(self):
        """Test file size calculation"""
        # Create a temporary file
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            test_file = temp_dir / "test.txt"
            test_file.write_text("x" * 1024 * 1024)  # 1 MB
            
            size = get_file_size_mb(test_file)
            assert size == pytest.approx(1.0, rel=0.1)
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
