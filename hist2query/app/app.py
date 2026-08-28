from typing import Tuple
import pickle
import os
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
import torch
import timm
import polars as pl
import faiss
import torch.nn.functional as tf
import torchvision
from hist2query.app.utils import (
    TCGAUNI2QueryRequestParams,
    TCGA_RESPONSE_COL_HEADERS,
    preprocess_tiles,
    make_tiles,
    decode_patch,
    load_hf_model,
    load_prism2_processing,
    prism2_prompt_type, Prism2ChatRequestParams)

def create_app(model_loader: Callable[[], Tuple[
               timm.models.vision_transformer.VisionTransformer,
               torchvision.transforms.transforms.Compose, str]],
               index_loader: Callable[[], [faiss.Index]],
               metadata_loader: Callable[[], [pl.DataFrame]],
               enable_prism2: bool=False):

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        app.state.index = index_loader()
        app.state.metadata = metadata_loader()
        (app.state.uni2_model, app.state.prism2_model, app.state.uni2_transform, app.state.prism2_processor,
        app.state.virchow2_model, app.state.virchow2_transform) = None, None, None, None, None, None
        app.state.device = "cuda" if torch.cuda.is_available() else "cpu"

        if app.state.index is not None and app.state.metadata is not None:

            # only load uni2 if the index and metadata are present
            app.state.uni2_model, app.state.uni2_transform, app.state.device = model_loader()
            # allow uni2 to be on GPU with Prism2
            app.state.uni2_model.to(app.state.device)

        if enable_prism2 and torch.cuda.is_available():

            # keep Virchow2 on CPU to avoid having too many models on GPU
            app.state.virchow2_model, app.state.virchow2_transform, app.state.device = load_hf_model("hf-hub:paige-ai/Virchow2")
            app.state.virchow2_model.to("cpu")
            app.state.prism2_model, app.state.prism2_processor = load_prism2_processing()
            app.state.prism2_model.eval()
            app.state.prism2_model.to(app.state.device)

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "tcga_uni_slide_filenames.pkl"), "rb") as slide_names_open:
            # use the pkl as a package data file and map the slide URLs to avoid nested per request GDC portal API calls
            app.state.tcga_uni_slide_filenames = pickle.load(slide_names_open)

        app.state.inference_lock = asyncio.Lock()

        yield

    app = FastAPI(title="hist2query", lifespan=lifespan)

    @app.post("/search")
    async def search(
            patch: UploadFile = File(...),
            params: TCGAUNI2QueryRequestParams = Depends(TCGAUNI2QueryRequestParams.as_form)):

        if any(elem is None for elem in (app.state.index, app.state.metadata)):
            raise HTTPException(status_code=503,
                                detail="TCGA UNI2 queries not available: Either the FAISS index or matching parquet metadata file "
                                       "was not supplied to the hist2query deployment.")

        tile_rgb = await decode_patch(patch)

        batch = preprocess_tiles(make_tiles(tile_rgb))
        # a single transform per image is faster, but results seem notieceably worse
        # batch = app.state.uni2_transform(Image.fromarray(tile_rgb)).unsqueeze(dim=0)
        batch = batch.to(app.state.device)

        async with app.state.inference_lock:
            with torch.inference_mode():
                embedding = app.state.uni2_model(batch)

        # normalization must match what was used to create the index
        embedding = embedding.mean(dim=0)
        embedding = tf.normalize(embedding, p=2, dim=0)

        embedding = (embedding.unsqueeze(0).cpu().numpy().astype("float32"))

        scores, indices = app.state.index.search(embedding, params.k)

        del tile_rgb, batch, embedding

        resp = {'hits': None, 'url': None}
        indices_use = indices[0][indices[0] >= 0]
        scores_use = scores[0][indices[0] >= 0]
        results = app.state.metadata.filter(pl.col("index").is_in(indices_use)).collect()
        # sort the rows after collect to match the scores
        order = {idx: i for i, idx in enumerate(indices_use)}

        results = (results.with_columns(pl.col("index").replace(order).alias("_order"))
                   .sort("_order").drop("_order"))
        results = results.with_columns(pl.Series("similarity", scores_use))

        resp['hits'] = results.select(TCGA_RESPONSE_COL_HEADERS).to_dicts()
        # Add a URL per slide if requested, keep as separate key in the response to avoid redundant data packets
        if params.url:
            resp['url'] = {key: value for key, value in app.state.tcga_uni_slide_filenames.items()
                           if key in results['slide'].unique().to_list()}
        return resp


    @app.post("/chat")
    async def chat(patch: UploadFile = File(...),
                   params: Prism2ChatRequestParams = Depends(Prism2ChatRequestParams.as_form)):

        if not torch.cuda.is_available() or any(elem is None for elem in (app.state.virchow2_model, app.state.prism2_model)):
            raise HTTPException(status_code=503,
                detail="Prism2 not available: CUDA not found in the hist2query deployment, or "
                       "Prism2 was not enabled with `--use-prism2`.")

        tile_rgb = await decode_patch(patch)

        tile_rgb = make_tiles(tile_rgb)

        tile_batch = []
        # cannot use the same tile generation function as for uni2 as only the class token is pulled from Virchow2
        for tile_rgb in tile_rgb:
            output = app.state.virchow2_model(app.state.virchow2_transform(
                Image.fromarray(tile_rgb).convert("RGB")).unsqueeze(0))
            tile_batch.append(output[:, 0])

        tile_batch = torch.cat(tile_batch, dim=0)
        batch = app.state.prism2_processor([tile_batch]).to(app.state.device)
        async with app.state.inference_lock:
            with torch.autocast(app.state.device, torch.bfloat16):
                answers = prism2_prompt_type(app.state.prism2_model, str(params.question),
                                         batch, int(params.max_token_response),
                                         params.raw_scores_binary, params.binary_threshold_for_yes)
        del tile_rgb, tile_batch, batch
        return {'response': answers}

    return app
