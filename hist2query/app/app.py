from typing import Union, Tuple, Optional, Annotated
from pathlib import Path
import pickle
import io
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Depends
import torch
import timm
import faiss
import pandas as pd
import numpy as np
from pydantic import BaseModel
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as tf
import torchvision
from hist2query.app.utils import preprocess_tiles, make_tiles

TCGA_RESPONSE_COL_HEADERS = ['project', 'slide', 'x', 'y', 'similarity']

class SearchRequest(BaseModel):
    # set the default query parameters
    k: Optional[int] = 100
    url: Optional[bool] = True

    @classmethod
    def as_form(
        cls,
        k: Optional[int] = Form(100),
        url: Optional[bool] = Form(True)):

        return cls(k=k, url=url)

def load_uni_model(model_name: str="hf-hub:MahmoodLab/UNI2-h") -> \
        [timm.models.vision_transformer.VisionTransformer, torchvision.transforms.transforms.Compose, str]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        model_name,
        pretrained=True,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True)

    model.eval()
    model.to(device)

    transform = create_transform(**resolve_data_config(
            model.pretrained_cfg, model=model))

    return model, transform, device

def load_index(path: Union[str, Path]) -> faiss.Index:
    return faiss.read_index(path)

def load_metadata(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_parquet(path)

def create_app(model_loader: Callable[[], Tuple[timm.models.vision_transformer.VisionTransformer,
               torchvision.transforms.transforms.Compose, str]],
               index_loader: Callable[[], [faiss.Index]],
               metadata_loader: Callable[[], [pd.DataFrame]]):

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        app.state.model, app.state.transform, app.state.device = model_loader()

        app.state.index = index_loader()

        app.state.metadata = metadata_loader()

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tcga_uni_slide_filenames.pkl"), "rb") as slide_names_open:
            # use the pkl as a package data file and map the slide URLs to avoid nested per request GDC portal API calls
            app.state.tcga_uni_slide_filenames = pickle.load(slide_names_open)

        yield

    app = FastAPI(title="hist2query", lifespan=lifespan)

    @app.post("/search")
    async def search(
            patch: UploadFile = File(...),
            # search parameters are optional
            params: SearchRequest = Depends(SearchRequest.as_form)):

        patch_bytes = await patch.read()

        arr = np.load(io.BytesIO(patch_bytes))

        tile_rgb = np.array(arr["data"])

        tile_rgb = make_tiles(tile_rgb) if (tile_rgb.shape[0] > 224 or
                    tile_rgb.shape[1] > 224) else [tile_rgb]

        batch = preprocess_tiles(tile_rgb, app.state.transform)
        batch = batch.to(app.state.device)

        with torch.inference_mode():
            embedding = app.state.model(batch)

        # normalization must match what was used to create the index
        embedding = embedding.mean(dim=0)
        embedding = tf.normalize(embedding, p=2, dim=0)

        embedding = (embedding.unsqueeze(0).cpu().numpy().astype("float32"))

        scores, indices = app.state.index.search(embedding, params.k)

        resp = {'hits': None, 'url': None}
        results = app.state.metadata.iloc[indices[0]]
        results['similarity'] = scores[0]
        resp['hits'] = results[TCGA_RESPONSE_COL_HEADERS].to_dict(orient="records")
        # Add a URL per slide if requested, keep as separate key in the response to avoid redundant data packets
        if params.url:
            resp['url'] = {key: value for key, value in app.state.tcga_uni_slide_filenames.items()
                           if key in results['slide'].unique().tolist()}
        return resp

    return app