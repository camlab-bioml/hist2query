from typing import Union, Tuple
import pickle
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import torch
import timm
import faiss
import pandas as pd
import torch.nn.functional as tf
import torchvision
from hist2query.app.utils import (
    SearchRequest,
    TCGA_RESPONSE_COL_HEADERS,
    preprocess_tiles,
    make_tiles,
    decode_patch,
    load_hf_model, load_prism2_processing)

def create_app(model_loader: Callable[[], Tuple[
               timm.models.vision_transformer.VisionTransformer,
               torchvision.transforms.transforms.Compose, str]],
               index_loader: Callable[[], [faiss.Index]],
               metadata_loader: Callable[[], [pd.DataFrame]],
               enable_prism2: bool=False):

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        app.state.uni2_model, app.state.uni2_transform, app.state.device = model_loader()

        (app.state.prism2_model, app.state.prism2_processor, app.state.virchow2_model,
         app.state.virchow2_transform) = None, None, None, None

        if enable_prism2 and torch.cuda.is_available():

            app.state.virchow2_model, app.state.virchow2_transform, app.state.device = load_hf_model("hf-hub:paige-ai/Virchow2")
            app.state.prism2_model, app.state.prism2_processor = load_prism2_processing()
            app.state.prism2_model.eval()
            app.state.virchow2_model.to(app.state.device)
            app.state.prism2_processor.to(app.state.device)
            app.state.prism2_model.to(app.state.device)

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

        tile_rgb = await decode_patch(patch)

        tile_rgb = make_tiles(tile_rgb) if (tile_rgb.shape[0] > 224 or
                    tile_rgb.shape[1] > 224) else [tile_rgb]

        batch = preprocess_tiles(tile_rgb, app.state.uni2_transform)
        batch = batch.to(app.state.device)

        with torch.inference_mode():
            embedding = app.state.uni2_model(batch)

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


    @app.post("/chat")
    async def chat(patch: UploadFile = File(...),
                   question: Union[str, None]=None):

        if any(elem is None for elem in (app.state.prism2_model, app.state.virchow2_model)):
            raise HTTPException(status_code=503,
                detail="Prism2 not available: CUDA not found in the hist2query deployment.")

        tile_rgb = await decode_patch(patch)

        image = app.state.virchow2_model(app.state.virchow2_transform(
                Image.fromarray(tile_rgb).convert('RGB')).unsqueeze(
                0).to(app.state.device)).to(app.state.device)

        batch = app.state.prism2_processor(image[:, 0]).to(app.state.device)
        with torch.autocast(app.state.device, torch.bfloat16):
            answers = app.state.prism2_model.get_response(**batch,
                prompt=str(question), max_new_tokens=100)

        return {'response': answers}

    return app