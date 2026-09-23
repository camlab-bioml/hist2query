import numpy as np
from hist2query.utils import (
    set_tcga_projects_include,
    TCGA_STUDY_CODES)
from hist2query.app.utils import make_tiles, extract_virchow2_embeddings
from conftest import MockVirchow2Model, MockUNITransform

def test_make_tiles():
    tiles = make_tiles(np.random.randint(0, 255, size=(448, 448, 3)))
    # should be a tile per 224x224 pixel patch
    assert tiles.shape[0] == 4
    missing_edges = make_tiles(np.random.randint(0, 255, size=(700, 700, 3)))
    assert missing_edges.shape[0] == 9
    stride_larger = make_tiles(np.random.randint(0, 255, size=(100, 100, 3)))
    assert stride_larger.shape[0] == 0

    stride_smaller = make_tiles(np.random.randint(0, 255, size=(224, 224, 3)), stride=100)
    assert stride_smaller.shape[0] == 1

    # gives positions, (0, 0), (0, 100), (100, 0), (100, 100)
    stride_smaller_larger_image = make_tiles(
        np.random.randint(0, 255, size=(400, 400, 3), dtype=np.uint8),
        tile_size=224,
        stride=100)
    assert stride_smaller_larger_image.shape[0] == 4

    no_tiles = make_tiles(np.random.randint(0, 255, size=(1, 1, 3)), stride=100)
    assert no_tiles.shape[0] == 0

def test_virchow2_tile_generation():
    tiles = make_tiles(np.random.randint(0, 255, size=(448, 448, 3)))
    virchow2_embeddings = extract_virchow2_embeddings(MockVirchow2Model(), MockUNITransform(), tiles)
    assert virchow2_embeddings.shape == (4, 1280)
    assert extract_virchow2_embeddings(MockVirchow2Model(), MockUNITransform(), []) is None

def test_set_tcga_projects():

    assert set_tcga_projects_include() == list(TCGA_STUDY_CODES.keys())

    include_by_proj_name = set_tcga_projects_include("READ,UCS,MESO")
    assert include_by_proj_name == ['TCGA-MESO', 'TCGA-READ', 'TCGA-UCS']
    include_by_tissue_partial = set_tcga_projects_include("Bladder,Breast,Lung")
    assert include_by_tissue_partial == ['TCGA-BLCA', 'TCGA-BRCA_IDC', 'TCGA-BRCA_OTHERS', 'TCGA-LUAD', 'TCGA-LUSC']

    include_mix_project_tissue = set_tcga_projects_include("UCEC,Thyroid,SARC,GBM,Colon")
    assert include_mix_project_tissue == ['TCGA-COAD', 'TCGA-GBM', 'TCGA-SARC', 'TCGA-THCA', 'TCGA-UCEC']

    exclude_by_project_name = set_tcga_projects_include(None, "READ,UCS,MESO")
    for proj in exclude_by_project_name:
        assert not any(elem in proj for elem in ["READ", "UCS", "MESO"])

    exclude_by_tissue_name = set_tcga_projects_include(None, "Head,Liver,Ovarian")
    for proj in exclude_by_tissue_name:
        assert not any(elem in proj for elem in ["HNSC", "LIHC", "OV"])

    exclude_by_mixed = set_tcga_projects_include(None, "BLCA,Cervical,KIRP,Prostate,Uterine")
    for proj in exclude_by_mixed:
        assert not any(elem in proj for elem in ["BLCA", "CESC", "KIRP", "PRAD", "UCEC", "UCS"])
    assert len(exclude_by_mixed) == (len(TCGA_STUDY_CODES) - 6)

    set_both_include_exclude = set_tcga_projects_include("READ,UCS,MESO", "Head,Liver,Ovarian,MESO")
    assert len(set_both_include_exclude) == 2
    assert all('MESO' not in elem for elem in set_both_include_exclude)

    assert set_tcga_projects_include("TCGA-KIRP") == ['TCGA-KIRP']