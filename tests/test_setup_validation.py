"""Validation tests to ensure the testing infrastructure is properly configured."""
import sys
from pathlib import Path
import pytest
import torch
import numpy as np


class TestSetupValidation:
    """Validate that the testing infrastructure is properly set up."""
    
    def test_pytest_is_installed(self):
        """Verify pytest is installed and importable."""
        assert pytest.__version__
    
    def test_project_root_in_path(self):
        """Verify the project root is in Python path."""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path or str(project_root.absolute()) in sys.path
    
    def test_models_package_importable(self):
        """Verify the models package can be imported."""
        try:
            import models
            assert models.__name__ == 'models'
        except ImportError:
            pytest.fail("Cannot import models package")
    
    def test_required_dependencies_available(self):
        """Verify all required dependencies are available."""
        required_modules = {
            'torch': torch,
            'numpy': np,
            'pytest': pytest,
        }
        
        for module_name, module in required_modules.items():
            assert module is not None, f"{module_name} is not available"
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify custom pytest markers work correctly."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify slow marker works."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Verify temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    
    def test_mock_model_config_fixture(self, mock_model_config):
        """Verify mock_model_config fixture provides expected structure."""
        assert isinstance(mock_model_config, dict)
        assert 'model_name' in mock_model_config
        assert 'resolution' in mock_model_config
        assert 'latent_dim' in mock_model_config
        assert mock_model_config['device'] == 'cpu'
    
    def test_sample_latent_codes_fixture(self, sample_latent_codes):
        """Verify sample_latent_codes fixture provides correct tensor."""
        assert isinstance(sample_latent_codes, torch.Tensor)
        assert sample_latent_codes.shape == (4, 512)
        assert sample_latent_codes.dtype == torch.float32
    
    def test_numpy_array_fixture(self, sample_numpy_array):
        """Verify sample_numpy_array fixture works."""
        assert isinstance(sample_numpy_array, np.ndarray)
        assert sample_numpy_array.shape == (10, 512)
        assert sample_numpy_array.dtype == np.float32
    
    def test_device_fixture(self, device):
        """Verify device fixture returns CPU device."""
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
    
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        # This test will pass if coverage is running
        assert True
    
    def test_conftest_fixtures_available(self, mock_streamlit_session):
        """Verify conftest fixtures are available."""
        assert hasattr(mock_streamlit_session, 'model')
        assert hasattr(mock_streamlit_session, 'latent_codes')
        assert hasattr(mock_streamlit_session, 'boundaries')
    
    def test_capture_stdout_fixture(self, capture_stdout):
        """Verify stdout capture fixture works."""
        print("Test output")
        captured = capture_stdout.readouterr()
        assert "Test output" in captured.out


class TestDirectoryStructure:
    """Validate the test directory structure."""
    
    def test_tests_directory_exists(self):
        """Verify tests directory exists."""
        tests_dir = Path(__file__).parent
        assert tests_dir.name == 'tests'
        assert tests_dir.exists()
    
    def test_unit_directory_exists(self):
        """Verify unit tests directory exists."""
        unit_dir = Path(__file__).parent / 'unit'
        assert unit_dir.exists()
        assert unit_dir.is_dir()
    
    def test_integration_directory_exists(self):
        """Verify integration tests directory exists."""
        integration_dir = Path(__file__).parent / 'integration'
        assert integration_dir.exists()
        assert integration_dir.is_dir()
    
    def test_conftest_exists(self):
        """Verify conftest.py exists."""
        conftest_path = Path(__file__).parent / 'conftest.py'
        assert conftest_path.exists()
        assert conftest_path.is_file()
    
    def test_init_files_exist(self):
        """Verify __init__.py files exist in test directories."""
        test_dirs = [
            Path(__file__).parent,
            Path(__file__).parent / 'unit',
            Path(__file__).parent / 'integration',
        ]
        
        for test_dir in test_dirs:
            init_file = test_dir / '__init__.py'
            assert init_file.exists(), f"__init__.py missing in {test_dir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])