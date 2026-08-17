from collections.abc import Callable
from pathlib import Path
from typing import Optional, Any
from typing import Union
from pydantic import BaseModel
from fastapi import Form, UploadFile
import numpy as np
import pandas as pd
import torch
from PIL import Image
import timm
from timm.layers import SwiGLUPacked
import io
import faiss
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import httpx
from transformers import AutoModel, AutoProcessor

TCGA_RESPONSE_COL_HEADERS = ['project', 'tissue', 'slide', 'x', 'y', 'similarity']

class TCGAUNI2QueryRequestParams(BaseModel):
    # set the default query parameters
    k: Optional[int] = 100
    url: Optional[bool] = True

    @classmethod
    def as_form(
        cls,
        k: Optional[int] = Form(100),
        url: Optional[bool] = Form(True)):

        return cls(k=k, url=url)

class Prism2ChatRequestParams(BaseModel):
    # set the default query parameters
    question: Optional[str] = "Write a report."
    max_token_response: Optional[int] = 250
    raw_scores_binary: Optional[bool]= True
    binary_threshold_for_yes: Optional[float] = 0.5

    @classmethod
    def as_form(
        cls,
        question: Optional[str] = Form("Write a report."),
        max_token_response: Optional[int] = Form(250),
        raw_scores_binary: Optional[bool]=Form(False),
        binary_threshold_for_yes: Optional[float] = Form(0.5),):

        return cls(question=question,
                   max_token_response=max_token_response,
                   raw_scores_binary=raw_scores_binary,
                   binary_threshold_for_yes=binary_threshold_for_yes)

def make_tiles(img_patch: Union[np.ndarray, np.array],
               tile_size: int=224, stride: int=224) -> Union[list, None]:

    if len(img_patch.shape) < 3:
        return None
    img_patch = img_patch[:, :, :3] if img_patch.shape[2] != 3 else img_patch
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

    # TODO: should we pass the mean tensor here instead of a stack to the model?
    # appears comparable similarity results but somehow slower
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

HF_MODEL_KWARGS = {
"hf-hub:MahmoodLab/UNI2-h": {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True},
    "hf-hub:paige-ai/Virchow2":
    {'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU}}


def load_hf_model(model_name: str= "hf-hub:MahmoodLab/UNI2-h") -> \
        [timm.models.vision_transformer.VisionTransformer,
         torchvision.transforms.transforms.Compose, str]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        model_name,
        pretrained=True,
        **HF_MODEL_KWARGS[model_name])

    model.eval()
    model.to("cpu")
    
    transform = create_transform(**resolve_data_config(
            model.pretrained_cfg, model=model))

    return model, transform, device

def load_prism2_processing():

    return (AutoModel.from_pretrained("paige-ai/Prism2", trust_remote_code=True, torch_dtype=torch.bfloat16),
            AutoProcessor.from_pretrained("paige-ai/Prism2", trust_remote_code=True, torch_dtype=torch.bfloat16))

def load_index(path: Union[str, Path, None]=None) -> Union[faiss.Index, None]:
    """
    Read an FAISS index path if it exists, or return `None`
    """
    if not path: return None
    return faiss.read_index(path)

def load_metadata(path: Union[str, Path, None]=None) -> Union[pd.DataFrame, None]:
    """
    Read a metadata parquet file matching an FAISS index if the path exists, or return `None`
    """
    if not path: return None
    return pd.read_parquet(path)

async def decode_patch(patch: UploadFile):
    if str(patch.filename).endswith("png"):
        return np.asarray(Image.open(patch.file).convert("RGB"))
    else:
        patch_bytes = await patch.read()
        arr = np.load(io.BytesIO(patch_bytes))
        return np.array(arr["data"])

_PRISM2_YES_NO_IDENTIFIERS = ['are', 'is', 'can', 'could']

def binary_threshold_response(scores: torch.tensor,
                              prob_threshold: float=0.5):
    """
    Define the binary threshold for the yes/no prism2 response
    """
    return [f"Yes (P={round(float(score), 4)})" if score >= prob_threshold else
            f"No (P={round(float(score), 4)})" for score in scores]


def prism2_prompt_type(model: Any, question: str,
                       batch: torch.tensor,
                       max_tokens_response: int=250,
                       raw_scores_binary: bool=False,
                       binary_threshold_for_yes: Union[float, str]=0.5):
    """
    Detect the appropriate type of model to use (open-context vs. yes/no) and set the tokens accordingly
    """
    if any(str(question).lower().startswith(quest_starter) for quest_starter in _PRISM2_YES_NO_IDENTIFIERS):
        binary_resp = model.yes_no_score(
        tile_embeddings=batch["tile_embeddings"],
        attention_mask=batch["attention_mask"],
        question=str(question))

        return binary_threshold_response(binary_resp, float(binary_threshold_for_yes)) if not raw_scores_binary else \
            [round(float(elem), 4) for elem in binary_resp]

    return model.get_response(**batch,
                prompt=str(question), max_new_tokens=max_tokens_response)
