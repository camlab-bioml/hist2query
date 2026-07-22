from typing import Union
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from hist2query.process.download import (
    get_hf_projects, subsample_slides_by_project, process_tcga_slide)

TCGA_STUDY_CODES = {
    "TCGA-ACC": "Adrenocortical carcinoma",
    "TCGA-BLCA": "Bladder Urothelial Carcinoma",
    "TCGA-BRCA_IDC": "Bladder Urothelial Carcinoma",
    "TCGA-BRCA_OTHERS": "Breast invasive carcinoma",
    "TCGA-CESC": "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "TCGA-CHOL": "Cholangiocarcinoma",
    "TCGA-COAD": "Colon adenocarcinoma",
    "TCGA-DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    "TCGA-ESCA": "Esophageal carcinoma",
    "TCGA-GBM": "Glioblastoma multiforme",
    "TCGA-HNSC": "Head and Neck squamous cell carcinoma",
    "TCGA-KICH": "Kidney Chromophobe",
    "TCGA-KIRC": "Kidney renal clear cell carcinoma",
    "TCGA-KIRP": "Kidney renal papillary cell carcinoma",
    "TCGA-LGG": "Brain Lower Grade Glioma",
    "TCGA-LIHC": "Liver hepatocellular carcinoma",
    "TCGA-LUAD": "Lung adenocarcinoma",
    "TCGA-LUSC": "Lung squamous cell carcinoma",
    "TCGA-MESO": "Mesothelioma",
    "TCGA-OV": "Ovarian serous cystadenocarcinoma",
    "TCGA-PAAD": "Pancreatic adenocarcinoma",
    "TCGA-PCPG": "Pheochromocytoma and Paraganglioma",
    "TCGA-PRAD": "Prostate adenocarcinoma",
    "TCGA-READ": "Rectum adenocarcinoma",
    "TCGA-SARC": "Sarcoma",
    "TCGA-SKCM": "Skin Cutaneous Melanoma",
    "TCGA-STAD": "Stomach adenocarcinoma",
    "TCGA-TGCT": "Testicular Germ Cell Tumors",
    "TCGA-THCA": "Thyroid carcinoma",
    "TCGA-THYM": "Thymoma",
    "TCGA-UCEC": "Uterine Corpus Endometrial Carcinoma",
    "TCGA-UCS": "Uterine Carcinosarcoma",
    "TCGA-UVM": "Uveal Melanoma",
}

def add_to_index(repo_hf: str="W8Yi/tcga-wsi-uni2h-features",
    input_index: Union[str, Path, None]="tcga_uni_trained.index",
    output_index: Union[str, Path, None] = "tcga_uni_added.index",
    output_metadata: Union[str, Path, None] = "tcga_uni_metadata.parquet",
    slide_prop: Union[int, float]=0.25,
    patches_per_slide: int=100,
    min_slides_project: int=25,
    max_slides_project: int=75,
    workers: int=16,
    hf_token: Union[str, None]=None):

    index = faiss.read_index(input_index)
    sampled = subsample_slides_by_project(get_hf_projects(repo_hf), slide_prop,
                                          min_slides_project, max_slides_project)

    metadata_chunks = []
    embedding_index = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_tcga_slide, path, repo_hf, tmpdir,
                                   patches_per_slide, False, True, hf_token) for path in sampled]
            for future in as_completed(futures):
                if future is not None:
                    result = future.result()
                    if result is not None:
                        embeddings = result["embeddings"]

                        ids = np.arange(embedding_index,
                                        embedding_index + len(embeddings),
                                        dtype=np.int64)

                        index.add_with_ids(embeddings, ids)

                        metadata_chunks.append(
                            pd.DataFrame({
                                "index": ids,
                                "project": result["project"],
                                "tissue": TCGA_STUDY_CODES[result["project"]],
                                "slide": result["slide"],
                                "x": result["coords"][:, 0],
                                "y": result["coords"][:, 1],
                            }))

                        embedding_index += len(ids)

    pd.concat(metadata_chunks).to_parquet(output_metadata, index=False)

    faiss.write_index(index, output_index)