"""
Query the GDC API to get the diagnostic slide filenames for every TCGA UNI
slide. Allows mapping of the slide to the URL for viewing in the GDC data portal: https://portal.gdc.cancer.gov/
"""

import glob
from collections import defaultdict
import pickle
import asyncio
import os
from huggingface_hub import list_repo_files
from hist2query.app.utils import patient_url_gdc_portal

async def patient_slide_paths(slides_by_project: list):
    slide_paths_gdc = {}
    h5_file_base = [str(slide_tcga).split("/features/")[1] for slide_tcga in slides_by_project]
    for h5 in h5_file_base:
        slide_paths_gdc[h5] = await patient_url_gdc_portal(h5.split(".h5")[0])
    return slide_paths_gdc

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))

    files = list_repo_files(
        repo_id="W8Yi/tcga-wsi-uni2h-features",
        repo_type="dataset")

    h5_files = [f for f in files if f.endswith(".h5") and "features" in f]

    projects = defaultdict(list)

    for f in h5_files:
        project = f.split("/")[0]
        projects[project].append(f)

    for proj, slides in projects.items():
        out_filename = f"{current_dir}/{proj}_slide_filenames.pkl"
        # create a pickle file per project, in case of API request overloads or resume
        if not os.path.isfile(out_filename):
            slide_paths = asyncio.run(patient_slide_paths(slides))
            with open(out_filename, "wb") as file:
                pickle.dump(slide_paths, file)

    slide_hash = glob.glob(f"{current_dir}/TCGA*_slide_filenames.pkl")

    # merge project-specific file name stores into one
    merged = {}
    for slide in slide_hash:
        with open(slide, 'rb') as file:
            tcga_slides = pickle.load(file)
            merged = merged | tcga_slides

    with open(os.path.join(current_dir, "all_slide_filenames.pkl"), "wb") as file:
        pickle.dump(merged, file)
