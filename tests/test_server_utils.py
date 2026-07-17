from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pytest
from hist2query.app.utils import (
    make_tiles,
    patient_url_gdc_portal)

def test_tile_processing():
    tiles = np.ones((224* 3, 224*3, 3))
    tile_stack = make_tiles(tiles)
    assert len(tile_stack) == 9

    tiles = np.ones((900, 900, 3))
    tile_stack = make_tiles(tiles)
    assert len(tile_stack) == 16

    assert make_tiles(np.ones((224* 3, 224*3))) is None

@pytest.mark.asyncio
@patch("hist2query.app.utils.httpx.AsyncClient")
async def test_gdc_portal_slide_url(mock_client):
    expected = "https://portal.gdc.cancer.gov/files/e826d31a-01db-4489-afa4-95366385302b"

    mock_response = Mock()

    mock_response.json.return_value = {
        "data": {"hits": [
                {"file_id": "e826d31a-01db-4489-afa4-95366385302b",
                    "file_name": "TCGA-12-0670-01Z-00-DX3.h5",
                    "data_format": "SVS"}]}}

    # async context manager mock
    mock_client_instance = AsyncMock()

    mock_client_instance.post.return_value = mock_response

    mock_client.return_value.__aenter__.return_value = (
        mock_client_instance)

    result = await patient_url_gdc_portal("TCGA-12-0670-01Z-00-DX3.h5")

    assert result == expected

    mock_client_instance.post.assert_called_once()

    result_different = await patient_url_gdc_portal("TCGA-12-0670-01Z-00-DX2.h5")
    assert result_different is None
