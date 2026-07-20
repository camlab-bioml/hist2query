import io
from unittest.mock import patch, Mock
from typing import Union
import numpy as np
import pandas as pd
from hist2query.app.app import load_uni_model
from hist2query.cli.serve import run

def serialize_crop(crop: Union[np.array, np.ndarray, None]=None):
    if crop is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=np.stack(crop))
        return buffer.getvalue()
    return None

@patch("hist2query.app.app.create_transform")
@patch("hist2query.app.app.timm.create_model")
def test_load_uni_model(mock_create_model, mock_create_transform):

    class MockModel:
        pretrained_cfg = {}

        def eval(self):
            return self

        def to(self, device):
            return self

    mock_create_model.return_value = MockModel()
    mock_create_transform.return_value = "mock_transform"

    model, transform, device = load_uni_model()

    assert device in ["cpu", "cuda"]
    assert transform == "mock_transform"

    mock_create_model.assert_called_once()
    mock_create_transform.assert_called_once()

def test_tcga_uni_search_post(client):

    payload = serialize_crop(np.random.randint(0, 255,
        size=(224, 224, 3), dtype=np.uint8))

    response = client.post("/search",
        files={"patch": ("patch.npy",
            payload, "application/octet-stream")},
        data={"k": "10", "url": True})

    assert response.status_code == 200
    response_data = response.json()
    assert 'hits' in response_data
    assert 'url' in response_data
    assert len(response_data['hits']) == 10
    assert all('project' in elem for elem in response_data['hits'])
    assert all(slide in pd.DataFrame(response_data['hits'])['slide'].unique().tolist()
               for slide in list(response_data['url'].keys()) )

    response_no_url = client.post("/search",
            files={"patch": ("patch.npy",
            payload, "application/octet-stream")},
            data={"k": "50", "url": False})

    response_data = response_no_url.json()
    assert len(response_data['hits']) == 50
    assert response_data['url'] is None

    response_no_params = client.post("/search",
        files={"patch": ("patch.npy", payload, "application/octet-stream")},
        data=None)

    response_data = response_no_params.json()
    assert len(response_data['hits']) == 100
    assert 'url' in response_data

@patch("hist2query.cli.serve.uvicorn.run")
@patch("hist2query.cli.serve.create_app")
def test_app_cli(mock_app, mock_uvicorn):
    args = Mock()

    args.model = "fake-model"
    args.index = "fake.index"
    args.metadata = "fake.parquet"
    args.host = "127.0.0.1"
    args.port = 8000
    args.workers = 1

    fake_app = Mock()
    mock_app.return_value = fake_app

    run(args)

    mock_app.assert_called_once()
    mock_uvicorn.assert_called_once_with(
        fake_app,
        host="127.0.0.1",
        port=8000,
        workers=1)