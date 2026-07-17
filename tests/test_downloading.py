import os
from unittest.mock import patch
from hist2query.process.download import (
    get_hf_projects,
    process_tcga_slide,
    subsample_slides_by_project)

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

@patch("hist2query.process.download.download_hf_file")
def test_processing_tcga_slide(mock_tcga_slide, get_current_dir):

    mock_tcga_slide.return_value = ('TCGA-VD-A8KA-01Z-00-DX1.h5', os.path.join(get_current_dir, 'fixtures',
                                                'TCGA-VD-A8KA-01Z-00-DX1.h5'))

    slide_info = process_tcga_slide("TCGA_BRCA/features/TCGA-VD-A8KA-01Z-00-DX1.h5", "fake/repo",
                                    "fake_outdir", 200, False, False)
    assert slide_info['project'] == "TCGA_BRCA"
    assert slide_info['embeddings'].shape == (200, 1536)
    assert slide_info['coords'].shape == (200, 2)

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