import os
from unittest.mock import patch
import numpy as np
from hist2query.process.download import (
    get_hf_projects,
    process_tcga_slide,
    subsample_slides_by_project, download_hf_file)

@patch("hist2query.process.download.list_repo_files")
def test_tcga_uni_projects(mock_list_repo_files):

    mock_list_repo_files.return_value = [
        "TCGA-BRCA/features/slide1.h5",
        "TCGA-BRCA/features/slide2.h5",
        "TCGA-LUAD/features/slide3.h5",
        "TCGA-LUAD/features/slide4.h5",
        "metadata.csv",
        "README.md"]

    project_files = get_hf_projects("fake/repo")
    assert 'TCGA-BRCA' in project_files.keys()
    assert len(project_files) == 2
    assert 'TCGA-UVM' not in project_files.keys()

@patch("hist2query.process.download.hf_hub_download")
def test_basic_download(mock_hf_download, get_current_dir):
    mock_hf_download.return_value =  os.path.join(get_current_dir, 'fixtures',
                                                'TCGA-VD-A8KA-01Z-00-DX1.h5')

    hf_path, local_path = download_hf_file('TCGA-VD-A8KA-01Z-00-DX1.h5')
    assert hf_path == 'TCGA-VD-A8KA-01Z-00-DX1.h5'
    assert local_path == os.path.join(get_current_dir, 'fixtures',
                                                'TCGA-VD-A8KA-01Z-00-DX1.h5')

@patch("hist2query.process.download.hf_hub_download")
def test_basic_download_empty(mock_hf_download, get_current_dir):
    mock_hf_download.return_value = None
    mock_hf_download.side_effect = OSError("Invalid data")
    assert download_hf_file('TCGA-VD-A8KA-01Z-00-DX1.h5') is None

def test_processing_tcga_slide(get_current_dir):

    slide_info = process_tcga_slide(os.path.join(get_current_dir, 'fixtures',
                                                'TCGA-VD-A8KA-01Z-00-DX1.h5'), 'TCGA-VD-A8KA-01Z-00-DX1',
                                    'TCGA_BRCA', 200, False)
    assert isinstance(slide_info, dict)
    assert slide_info['project'] == "TCGA_BRCA"
    assert slide_info['embeddings'].shape == (200, 1536)
    assert slide_info['coords'].shape == (200, 2)

def test_processing_tcga_slide_training(get_current_dir):

    slide_info = process_tcga_slide(os.path.join(get_current_dir, 'fixtures',
                                                 'TCGA-VD-A8KA-01Z-00-DX1.h5'), 'TCGA-VD-A8KA-01Z-00-DX1',
                                    'TCGA_BRCA', 200, True)
    assert isinstance(slide_info, np.ndarray)
    assert slide_info.shape == (200, 1536)

def test_processing_tcga_slide_malformed(get_current_dir):

    # mock: if the download doesn't return a valid download, then no processing happens
    assert process_tcga_slide(None, "fake/repo",
                                    "fake_outdir", 200, False) is None

def test_downsample_project_slide_lists():
    slide_lists = {"TCGA_BRCA": ["slide_brca"] * 400,
                   "TCGA_UVM": ["slide_uvm"] * 40,
                   "TCGA_LUAD": ["slide_luad"] * 200}
    to_sample = subsample_slides_by_project(slide_lists, slide_prop=0.4, max_slides_project=500,
                                            min_slides_project=1)
    assert len(to_sample) == 256

    get_all = subsample_slides_by_project(slide_lists, slide_prop=None, max_slides_project=500,
                                            min_slides_project=1)
    assert len(get_all) == 640