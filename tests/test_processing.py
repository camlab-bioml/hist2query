import tempfile
from unittest.mock import patch
import os
import faiss
import h5py
import numpy as np
import pandas as pd
import pytest
from hist2query.process.train import train_index
from hist2query.process.add import add_to_index

@patch("hist2query.process.train.process_tcga_slide")
@patch("hist2query.process.train.subsample_slides_by_project")
@patch("hist2query.process.train.download_hf_file")
# order of arguments to order of patch decorators is reversed
def test_train_processing(mock_hf_download, mock_slides_project, mock_slide_process,
                          get_current_dir):

    mock_hf_download.return_value = ('TCGA-BRCA_OTHERS/features/TCGA-VD-A8KA-01Z-00-DX1.h5', os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'))
    with h5py.File(os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'), "r") as f:
        embeddings = np.squeeze(f["features"][:]).astype(np.float32)
        indices = np.random.choice(len(embeddings),
                                   min(200, len(embeddings)), replace=False)
        embeddings = embeddings[indices]
        mock_slide_process.return_value = embeddings

    mock_slides_project.return_value = ["slide_brca"] * 10 + ['slide_uvm'] * 20
    
    with tempfile.TemporaryDirectory() as tmp_test:
        train_index(output_index = os.path.join(tmp_test, 'test_build_train.index'),
                    nlist = 10, nbits = 4, patches_per_slide=500, min_slides_project=4, max_slides_project=20,
                    workers=2, remove_after_processing=False)
        assert os.path.isfile(os.path.join(tmp_test, 'test_build_train.index'))
        index = faiss.read_index(str(os.path.join(tmp_test, 'test_build_train.index')))
        assert index.d == 1536
        assert index.is_trained

@patch("hist2query.process.train.process_tcga_slide")
@patch("hist2query.process.train.subsample_slides_by_project")
@patch("hist2query.process.train.download_hf_file")
def test_train_processing_malformed(mock_hf_download, mock_slides_project, mock_slide_process,
                          get_current_dir):

    # if the input from the download is not compatible (wrong file type or missing tuple elements)
    mock_hf_download.return_value = (os.path.join(get_current_dir, 'fixtures', 'test_index_added.index'))

    mock_slide_process.return_value = None
    mock_slides_project.return_value = [os.path.join(get_current_dir, 'fixtures', 'test_index_added.index')]

    with tempfile.TemporaryDirectory() as tmp_test:
        # expect no arrays to be concatenated
        with pytest.raises(ValueError):
            train_index(output_index=os.path.join(tmp_test, 'test_build_train.index'),
                    nlist=10, nbits=4, patches_per_slide=500, min_slides_project=4, max_slides_project=20,
                    workers=2, remove_after_processing=False)

@patch("hist2query.process.train.process_tcga_slide")
@patch("hist2query.process.train.subsample_slides_by_project")
@patch("hist2query.process.train.download_hf_file")
def test_train_processing_empty(mock_hf_download, mock_slides_project, mock_slide_process,
                          get_current_dir):
    mock_hf_download.return_value = ('TCGA-BRCA_OTHERS/features/TCGA-VD-A8KA-01Z-00-DX1.h5',
                                     os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'))

    mock_slide_process.return_value = None
    mock_slides_project.return_value = ["slide_brca"] * 10 + ['slide_uvm'] * 20

    with tempfile.TemporaryDirectory() as tmp_test:
        # expect no arrays to be concatenated
        with pytest.raises(ValueError):
            train_index(output_index=os.path.join(tmp_test, 'test_build_train.index'),
                    nlist=10, nbits=4, patches_per_slide=500, min_slides_project=4, max_slides_project=20,
                    workers=2, remove_after_processing=False)

@patch("hist2query.process.add.process_tcga_slide")
@patch("hist2query.process.add.subsample_slides_by_project")
@patch("hist2query.process.add.download_hf_file")
def test_add_processing(mock_hf_download, mock_slides_project, mock_slide_process,
                          get_current_dir):
    
    mock_hf_download.return_value = ('TCGA-BRCA_OTHERS/features/TCGA-VD-A8KA-01Z-00-DX1.h5', os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'))
    with h5py.File(os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'), "r") as f:
        embeddings = np.squeeze(f["features"][:]).astype(np.float32)
        coords = np.squeeze(f["coords"][:]).astype(np.uint32)
        indices = np.random.choice(len(embeddings),
                                   min(500, len(embeddings)), replace=False)
        embeddings = embeddings[indices]
        mock_slide_process.return_value = {"project": "TCGA-BRCA_OTHERS",
                                           "slide": 'TCGA-VD-A8KA-01Z-00-DX1.h5',
                                           "tissue": "Breast invasive carcinoma (Other)",
                    "embeddings": embeddings, "coords": coords[indices]}

    mock_slides_project.return_value = ["TCGA-BRCA_OTHERS"]

    index_in = os.path.join(get_current_dir, 'fixtures', 'test_index_added.index')
    index_read = faiss.read_index(index_in)
    num_vectors = index_read.ntotal
    with tempfile.TemporaryDirectory() as tmp_test:
        index_out = os.path.join(tmp_test, 'test_index_added_more.index')
        metadata_out = os.path.join(tmp_test, 'test_more_added.parquet')
        add_to_index(input_index = index_in, output_index = index_out,
                     output_metadata = metadata_out, workers=2,
                    patches_per_slide=500, min_slides_project=1, max_slides_project=2,
                    remove_after_processing=False)
        assert os.path.isfile(os.path.join(tmp_test, 'test_index_added_more.index'))
        assert os.path.isfile(os.path.join(tmp_test, 'test_more_added.parquet'))
        index_back = faiss.read_index(os.path.join(tmp_test, 'test_index_added_more.index'))
        assert index_back.d > 0
        assert index_back.ntotal > num_vectors
        metadata_in = pd.read_parquet(os.path.join(tmp_test, 'test_more_added.parquet'))
        assert len(metadata_in) == 500
        assert metadata_in['project'].unique().tolist() == ['TCGA-BRCA_OTHERS']
        assert metadata_in['slide'].unique().tolist() == ['TCGA-VD-A8KA-01Z-00-DX1.h5']
        assert metadata_in['tissue'].unique().tolist() == ['Breast invasive carcinoma (Other)']


@patch("hist2query.process.add.process_tcga_slide")
@patch("hist2query.process.add.subsample_slides_by_project")
@patch("hist2query.process.add.download_hf_file")
def test_add_processing_empty(mock_hf_download, mock_slides_project, mock_slide_process,
                        get_current_dir):
    mock_hf_download.return_value = ('TCGA-BRCA_OTHERS/features/TCGA-VD-A8KA-01Z-00-DX1.h5',
                                     os.path.join(get_current_dir, 'fixtures', 'TCGA-VD-A8KA-01Z-00-DX1.h5'))

    mock_slide_process.return_value = None
    mock_slides_project.return_value = ["TCGA-BRCA_OTHERS"]

    index_in = os.path.join(get_current_dir, 'fixtures', 'test_index_added.index')
    index_read = faiss.read_index(index_in)
    num_vectors = index_read.ntotal
    with tempfile.TemporaryDirectory() as tmp_test:
        index_out = os.path.join(tmp_test, 'test_index_added_more.index')
        metadata_out = os.path.join(tmp_test, 'test_more_added.parquet')
        add_to_index(input_index=index_in, output_index=index_out,
                     output_metadata=metadata_out, workers=2,
                     patches_per_slide=500, min_slides_project=1, max_slides_project=2,
                     remove_after_processing=False)
        assert os.path.isfile(os.path.join(tmp_test, 'test_index_added_more.index'))
        assert not os.path.isfile(os.path.join(tmp_test, 'test_more_added.parquet'))
        index_back = faiss.read_index(os.path.join(tmp_test, 'test_index_added_more.index'))
        assert index_back.ntotal == num_vectors