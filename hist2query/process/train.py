from typing import Union
from pathlib import Path
import numpy as np
import faiss
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from hist2query.process.download import process_tcga_slide, get_hf_projects, subsample_slides_by_project

def train_index(
    repo_hf: str="W8Yi/tcga-wsi-uni2h-features",
    output_index: Union[str, Path, None]="tcga_uni_trained.index",
    nlist: int=8192,
    m_quantization: int=96,
    nbits: int=8,
    slide_prop: Union[int, float]=0.25,
    patches_per_slide: Union[int, None]=100,
    min_slides_project: Union[int, None]=25,
    max_slides_project: Union[int, None]=75,
    workers: int=16,
    hf_token: Union[str, None]=None):

    sampled = subsample_slides_by_project(get_hf_projects(repo_hf), slide_prop,
                                          min_slides_project, max_slides_project)

    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_tcga_slide, path, repo_hf, tmpdir,
                        patches_per_slide, True, True, hf_token) for path in sampled]
            train_embeddings = [future.result() for future in as_completed(futures) if
                                (future is not None and future.result() is not None)]

    train_embeddings = np.concatenate(train_embeddings)
    faiss.normalize_L2(train_embeddings)
    quantizer = faiss.IndexFlatIP(train_embeddings.shape[1])

    index = faiss.IndexIVFPQ(quantizer, train_embeddings.shape[1], nlist, m_quantization,
        nbits, faiss.METRIC_INNER_PRODUCT)

    index.train(train_embeddings)

    faiss.write_index(index, output_index)
