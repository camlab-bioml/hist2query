from collections.abc import Callable
from typing import Union
import numpy as np
import torch
from PIL import Image
import httpx

def make_tiles(img_patch: Union[np.ndarray, np.array],
               tile_size: int=224, stride: int=224) -> Union[list, None]:
    # TODO: better checks for RGB shapes?
    if len(img_patch.shape) < 3 or img_patch.shape[2] != 3:
        return None
    H, W, _ = img_patch.shape
    tiles = []
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            new_tile = img_patch[y:y+tile_size, x:x+tile_size]
            tiles.append(new_tile)
    return tiles

def preprocess_tiles(tiles: list, transform: Callable) -> torch.tensor:

    processed = []

    for tile in tiles:
        img = Image.fromarray(tile.astype(np.uint8))
        img = transform(img)
        processed.append(img)

    return torch.stack(processed)

async def patient_url_gdc_portal(slide_id: str) -> Union[str, None]:
    """
    Get the patient URL of the slide from the GCD portal
    """

    query = {"filters": {"op": "in",
            "content": {"field": "cases.submitter_id",
                "value": ["-".join(slide_id.split("-")[:3])]}},
        "fields": "file_id,file_name,data_type,data_format,cases.submitter_id",
        "format": "JSON", "size": 100}

    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.gdc.cancer.gov/files", json=query)

    for file in r.json()['data']['hits']:
        # IMP: a patient can have multiple slides, querying without barcode in file name will give all patient slides
        # to limit to a slide, check that the barcode is part of the file name
        if 'SVS' in file['data_format'] and slide_id in file['file_name']:
            # if 'SVS' in file['data_format']:
            # this URL directs to the interactive slide viewer for that patient slide
            return f"https://portal.gdc.cancer.gov/files/{file['file_id']}"
    return None