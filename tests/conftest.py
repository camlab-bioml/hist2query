import os
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
from hist2query.app.app import (
    create_app,
    load_index,
    load_metadata)

@pytest.fixture(scope="session")
def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))

class MockUniModel:

    def __call__(self, x):
        return torch.ones((len(x),1536))

class MockUNITransform:

    def __call__(self, img):

        arr = np.array(img).astype(np.float32)

        # H,W,C -> C,H,W
        arr = torch.from_numpy(
            arr.transpose(2,0,1))

        # scale like torchvision ToTensor()
        arr = arr / 255.0

        # fake ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)

        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        # mock a normalization
        return (arr - mean) / std

@pytest.fixture(scope="session")
def mock_model_loader():
    def loader():
        return MockUniModel(), MockUNITransform(), "cpu"
    return loader

@pytest.fixture(scope="session")
def mock_index_loader(get_current_dir):
    def loader():
        return load_index(os.path.join(get_current_dir, 'fixtures', 'test_index_added.index'))
    return loader

@pytest.fixture(scope="session")
def mock_metadata_loader(get_current_dir):
    def loader():
        return load_metadata(os.path.join(get_current_dir, 'fixtures', 'test_metadata.parquet'))
    return loader

@pytest.fixture(scope="session")
def client(mock_model_loader, mock_index_loader, mock_metadata_loader):

    app = create_app(mock_model_loader, mock_index_loader, mock_metadata_loader)

    with TestClient(app) as client:
        yield client