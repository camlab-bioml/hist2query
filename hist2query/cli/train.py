from functools import partial
from hist2query.process.train import train_index
from hist2query.utils import numerical_or_none

def configure_parser(parser):
    parser.add_argument('-r', "--repo", dest="repo", default="W8Yi/tcga-wsi-uni2h-features",
        help="UNI model name or path")

    parser.add_argument('-o', "--index-out", dest="index", required=True,
        help="Out path to the index to be generated.")

    parser.add_argument('-nc', "--nlist", type=int, default=8192, dest="nlist",
        help="Number of clusters in the FAISS.")

    parser.add_argument('-m', "--mquant", type=int, default=96, dest="m_value",
        help="Number of sub-vectors for encoding per embedding.")

    parser.add_argument('-nb', "--nbits", default=8, type=int, dest="nbits",
        help="Number of bytes to encode each embedding.")

    parser.add_argument('-sp', "--slide-prop", default=0.25, type=partial(numerical_or_none, numerical_type=float), dest="slide_prop",
                        help="Set a proportion of each project to train on. If None, train on everything (NOT recommended).")

    parser.add_argument('-p', "--patches-per-slide", default=100, type=partial(numerical_or_none, numerical_type=int), dest="patches_per_slide",
                        help="Number of patches per slide to sub-sample and train on. If None, train on all patches (NOT recommended).")

    parser.add_argument('-minsp', "--min-slides-project", default=25, type=partial(numerical_or_none, numerical_type=int), dest="min_slides_project",
                        help="Minimum number of slides to include in training per project. If None, all slides per project are used (NOT recommended).")

    parser.add_argument('-maxsp', "--max-slides-project", default=75, type=partial(numerical_or_none, numerical_type=int), dest="max_slides_project",
                        help="Maximum number of slides to include in training per project. If None, all slides per project are used (NOT recommended).")

    parser.add_argument('-w', "--workers", default=16, type=int, dest="workers",
        help="Number of pool workers to use for multi-threaded hf downloads.")

    parser.add_argument('-t', "--hf-token", dest="token", type=str, default=None,
        help="Hugging Face access token.")

    parser.add_argument('-in', "--types-include", dest="types_include", type=str, default=None,
                        help="Optional list of project or tissue types to include. Should be separated by commas"
                             "i.e. Kidney,Prostate,Ovary or KIRP,PRAD,OV. If not specified, all project and tissue types are used.")

    parser.add_argument('-ex', "--types-exclude", dest="types_exclude", type=str, default=None,
                        help="Optional list of project or tissue types to exclude. Should be separated by commas"
                             "i.e. Kidney,Prostate,Ovary or KIRP,PRAD,OV. If not specified, all project and tissue types are used.")

    parser.set_defaults(func=run)


def run(args):

    train_index(args.repo, args.index, args.nlist, args.m_value, args.nbits,
                args.slide_prop, args.patches_per_slide, args.min_slides_project,
                args.max_slides_project, args.workers, args.token, True, args.types_include, args.types_exclude)