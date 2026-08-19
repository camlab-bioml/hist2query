from hist2query.utils import (
    set_tcga_projects_include,
    TCGA_STUDY_CODES)

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