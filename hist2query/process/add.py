from typing import Union
import tempfile
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import faiss
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from hist2query.process.download import (
    get_hf_projects,
    subsample_slides_by_project,
    process_tcga_slide,
    download_hf_file)

TCGA_STUDY_CODES = {
    "TCGA-ACC": "Adrenocortical carcinoma",
    "TCGA-BLCA": "Bladder Urothelial Carcinoma",
    "TCGA-BRCA_IDC": "Breast invasive carcinoma (IDC)",
    "TCGA-BRCA_OTHERS": "Breast invasive carcinoma (Other)",
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
    "TCGA-UVM": "Uveal Melanoma"}

def add_to_index(repo_hf: str="W8Yi/tcga-wsi-uni2h-features",
    input_index: Union[str, Path, None]="tcga_uni_trained.index",
    output_index: Union[str, Path, None] = "tcga_uni_added.index",
    output_metadata: Union[str, Path, None] = "tcga_uni_metadata.parquet",
    slide_prop: Union[int, float]=0.25,
    patches_per_slide: Union[int, None]=100,
    min_slides_project: Union[int, None]=25,
    max_slides_project: Union[int, None]=75,
    workers: int=16,
    hf_token: Union[str, None]=None,
    remove_after_processing: bool=True):

    index = faiss.read_index(input_index)
    sampled = subsample_slides_by_project(get_hf_projects(repo_hf), slide_prop,
                                          min_slides_project, max_slides_project)

    embedding_index = 0
    writer = None
    with tempfile.TemporaryDirectory() as tmpdir:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_hf_file, path, repo_hf, tmpdir,
                                   hf_token) for path in sampled]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    dl_path, local_path = None, None
                    if result is not None:
                        dl_path, local_path = result
                        sam_name, project_name = str(dl_path).split("/features/")[1], str(dl_path).split("/features/")[0]
                        result = process_tcga_slide(local_path, sam_name, project_name, patches_per_slide, False)
                        embeddings = result["embeddings"]

                        ids = np.arange(embedding_index,
                                        embedding_index + len(embeddings),
                                        dtype=np.int64)

                        index.add_with_ids(embeddings, ids)

                        df = pl.DataFrame({"index": ids,
                            "project": result["project"],
                            "slide": result["slide"],
                            "x": result["coords"][:, 0],
                            "y": result["coords"][:, 1],
                            }).with_columns(
                            pl.col("project")
                            .replace(TCGA_STUDY_CODES)
                            .alias("tissue"))

                        metadata_out = df.to_arrow()

                        if writer is None:
                            writer = pq.ParquetWriter(output_metadata, metadata_out.schema,
                                                      compression="snappy")

                        writer.write_table(metadata_out)

                        embedding_index += len(ids)
                        del result, embeddings, ids, df, metadata_out
                        gc.collect()
                except (OSError, KeyError, TypeError): pass
                finally:
                    if local_path is not None and os.path.exists(local_path) and remove_after_processing:
                        os.remove(local_path)
    
    if writer is not None: writer.close()

    faiss.write_index(index, output_index)