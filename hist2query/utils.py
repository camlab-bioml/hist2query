from typing import Union

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

def split_or_keep_str(string: str, delimiter: str=","):
    """
    Split a string if the delimiter is found, of simply return the element in a list
    """
    return str(string).split(str(delimiter)) if str(delimiter) in str(string) else [str(string)]

def set_tcga_projects_include(projects_include: Union[str,None]=None,
                              projects_exclude: Union[str,None]=None,
                              delimiter: str=","):
    """
    Set the TCGA projects to include for either index training or embedding addition.
    Combinations of inclusion and exclusion can be applied. Projects or tissue types can be supplied as
    either tissue names i.e. Kidney,Sarcoma or as project codes i.e. KIRP,SARC
    """

    projects_keep = []
    list_include = split_or_keep_str(projects_include, delimiter) if \
        projects_include is not None else list(TCGA_STUDY_CODES.keys())
    list_exclude = split_or_keep_str(projects_exclude, delimiter) if \
            projects_exclude is not None else []
    for project_name, study_code in TCGA_STUDY_CODES.items():
        if (any(elem in project_name for elem in list_include) or
           any(elem in study_code for elem in list_include)) and \
            (not any(elem in project_name for elem in list_exclude) and
            not any(elem in study_code for elem in list_exclude)):
            projects_keep.append(project_name)
    return projects_keep

def numerical_or_none(arg: Union[str, int, None],
                      numerical_type: type[int, float]=int):
    """
    Set an argparse arg as either None or a numerical type
    """
    if str(arg).lower() in ("none", "null") or arg is None:
        return None
    return numerical_type(arg)

def str_or_none(arg: Union[str, int, None]=None):
    """
    Set an argparse arg as either None or a numerical type
    """
    if str(arg).lower() in ("none", "null") or arg is None:
        return None
    return str(arg)