import os
import random
from typing import Union
from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np
import faiss
import h5py
from collections import defaultdict

def get_hf_projects(repo_hf: str="W8Yi/tcga-wsi-uni2h-features"):

    files_hf = list_repo_files(repo_id=repo_hf, repo_type="dataset")
    h5_files = [f for f in files_hf if f.endswith(".h5") and "features" in f]

    projects = defaultdict(list)

    for f in h5_files:
        project = f.split("/")[0]
        projects[project].append(f)

    return projects

def download_hf_file(dl_path: Union[str, None]=None,
                     repo_hf: Union[str, None]=None,
                     outdir: Union[str, None]=None,
                     hf_token: Union[str, None]=None):

    local_dl = hf_hub_download(repo_id=repo_hf, repo_type="dataset",
        filename=dl_path,local_dir=outdir,token=hf_token)
    return dl_path, local_dl

def process_tcga_slide(dl_path: Union[str, None]=None,
                     repo_hf: Union[str, None]=None,
                     outdir: Union[str, None]=None,
                     patches_per_slide: Union[int, None]=100,
                     training: bool=True,
                     remove_after_processing: bool=True,
                     hf_token: Union[str, None]=None):

    path_download = download_hf_file(dl_path, repo_hf, outdir, hf_token)[1]
    coords = None
    try:
        try:
            with h5py.File(path_download, "r") as f:
                embeddings = np.squeeze(f["features"][:]).astype(np.float32)
                if not training:
                    coords = np.squeeze(f["coords"][:])

            patches_per_slide = patches_per_slide if patches_per_slide is not None else len(embeddings)
            indices = np.random.choice(len(embeddings),
                                       min(patches_per_slide, len(embeddings)), replace=False)

            embeddings_hf = embeddings[indices]

            if training:
                return embeddings_hf

            sam_name = str(dl_path).split("/features/")[1]
            project_hf = str(dl_path).split("/features/")[0]

            faiss.normalize_L2(embeddings_hf)

            return {"project": project_hf, "slide": sam_name,
                    "embeddings": embeddings_hf, "coords": coords[indices]}

        except (OSError, TypeError):
            return None
    finally:
        if path_download is not None and os.path.isfile(path_download) and remove_after_processing:
            os.remove(path_download)


def subsample_slides_by_project(projects: dict,
                                slide_prop: Union[int, float, None]=0.25,
                                min_slides_project: Union[int, None]=25,
                                max_slides_project: Union[int, None]=75):
    sampled = []
    for project, slides in projects.items():
        # by default, use everything
        slide_count = len(slides)
        # if slide prop is between 0 and 1, subsample, and check the min and max slides per project
        if slide_prop is not None and 0 < slide_prop <= 1:
            min_slides_project = min_slides_project if min_slides_project is not None else len(slides)
            max_slides_project = max_slides_project if max_slides_project is not None else len(slides)
            slide_count = max(int(slide_prop * len(slides)), min_slides_project)
            slide_count = min(slide_count, max_slides_project)
        
        sampled.extend(random.sample(slides, min(slide_count, len(slides))))

    return sampled