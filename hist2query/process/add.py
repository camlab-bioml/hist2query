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
from hist2query.utils import TCGA_STUDY_CODES, set_tcga_projects_include


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
    remove_after_processing: bool=True,
    types_include: Union[str,None]=None,
    types_exclude: Union[str,None]=None):

    index = faiss.read_index(input_index)

    projects_default = get_hf_projects(repo_hf)
    project_names_keep = set_tcga_projects_include(types_include, types_exclude)
    projects_keep = {key: value for key, value in projects_default.items() if key in project_names_keep}

    sampled = subsample_slides_by_project(projects_keep, slide_prop,
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