"""Shared pytest fixtures and configuration for all tests."""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model_config() -> Dict[str, Any]:
    """Provide a mock model configuration for testing."""
    return {
        'model_name': 'test_model',
        'resolution': 256,
        'latent_dim': 512,
        'num_layers': 8,
        'device': 'cpu'
    }


@pytest.fixture
def sample_latent_codes() -> torch.Tensor:
    """Generate sample latent codes for testing."""
    batch_size = 4
    latent_dim = 512
    return torch.randn(batch_size, latent_dim)


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Generate a sample numpy array for testing."""
    return np.random.randn(10, 512).astype(np.float32)


@pytest.fixture
def mock_generator_output() -> torch.Tensor:
    """Mock output from a generator model."""
    batch_size = 4
    channels = 3
    height = width = 256
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing (CPU by default)."""
    return torch.device('cpu')


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with test files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create some sample files
    (data_dir / "latents.npy").touch()
    (data_dir / "config.json").touch()
    
    return data_dir


@pytest.fixture(autouse=True)
def reset_torch_seed():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for interface testing."""
    class MockSessionState:
        def __init__(self):
            self.model = None
            self.latent_codes = None
            self.boundaries = {}
            self.current_step = 0
    
    return MockSessionState()


@pytest.fixture(scope="session")
def test_models_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture
def cleanup_cuda():
    """Cleanup CUDA memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "gpu: mark test to run only when GPU is available"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test that requires pre-trained model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle special cases."""
    for item in items:
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="GPU not available")
            item.add_marker(skip_gpu)


@pytest.fixture
def capture_stdout(capsys):
    """Capture stdout for testing print statements."""
    yield capsys