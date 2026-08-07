import os
from unittest.mock import patch
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
from transformers import BatchEncoding
from hist2query.app.utils import (
    load_index,
    load_metadata)
from hist2query.app.app import create_app

@pytest.fixture(scope="session")
def get_current_dir():
    return str(os.path.abspath(os.path.join(os.path.dirname(__file__))))

class MockUniModel:

    def __call__(self, x):
        return torch.ones((len(x), 1536))

class MockVirchow2Model:

    def __call__(self, x):
        return torch.ones((1, len(x), 1280))

    def to(self, device: str):
        return self if device else self

class MockPrism2Model:

    def __call__(self, x):
        return {'batch': torch.ones((len(x), 2560))}

    def to(self, device: str):
        return self if device else self

    def eval(self):
        return self

    @staticmethod
    def get_response(*args, **kwargs):

        return "This is cancerous breast tissue"

class MockPrism2Transform:

    batch = None
    def __call__(self, img):
        # mock a tensor value that has a .to(device) property
        self.batch = BatchEncoding({"pixel_values": torch.from_numpy(
                np.array(img).astype(np.float32) / 255.0)})
        return self.batch

    def to(self, device: str):
        return self.batch if device else self.batch

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

        return (arr - mean) / std

@pytest.fixture(scope="session")
def mock_uni2_loader():
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
def client_no_prism2(mock_uni2_loader, mock_index_loader, mock_metadata_loader):

    app = create_app(mock_uni2_loader, mock_index_loader, mock_metadata_loader)

    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def client_prism2(mock_uni2_loader, mock_index_loader, mock_metadata_loader):
    with patch("hist2query.app.app.torch.cuda.is_available", return_value=True):
        with patch("hist2query.app.app.load_hf_model", return_value = (MockVirchow2Model(), MockUNITransform(), "cpu")):
            with patch("hist2query.app.app.load_prism2_processing", return_value = (MockPrism2Model(), MockPrism2Transform())):
                app = create_app(mock_uni2_loader, mock_index_loader, mock_metadata_loader, True)

                with TestClient(app) as client:
                    yield client