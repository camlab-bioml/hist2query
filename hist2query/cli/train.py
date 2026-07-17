from hist2query.process.train import train_index


def configure_parser(parser):
    parser.add_argument("--repo", dest="repo", default="W8Yi/tcga-wsi-uni2h-features",
        help="UNI model name or path")

    parser.add_argument("--index-out", dest="index", required=True,
        help="Out path to the index to be generated.")

    parser.add_argument("--nlist", type=int, default=8192, dest="nlist",
        help="Number of clusters in the FAISS.")

    parser.add_argument( "--mquant", type=int, default=96, dest="m_value",
        help="Number of sub-vectors for encoding per embedding.")

    parser.add_argument("--nbits", default=8, type=int, dest="nbits",
        help="Number of bytes to encode each embedding.")

    parser.add_argument("--slide-prop", default=0.25, type=float, dest="slide_prop",
        help="Set a proportion of each project to train on. If None, train on everything (NOT recommended).")

    parser.add_argument("--patches-per-slide", default=100, type=int, dest="patches_per_slide",
        help="Number of patches per slide to sub-sample and train on. If None, train on all patches (NOT recommended).")

    parser.add_argument("--min-slides-project", default=25, type=int, dest="min_slides_project",
        help="Minimum number of slides to include in training per project.")

    parser.add_argument("--max-slides-project", default=75, type=int, dest="max_slides_project",
        help="Maximum number of slides to include in training per project.")

    parser.add_argument("--workers", default=16, type=int, dest="workers",
        help="Number of pool workers to use for multi-threaded hf downloads.")

    parser.add_argument("--hf-token", dest="token", type=str, default=None,
        help="Hugging Face access token.")

    parser.set_defaults(func=run)


def run(args):

    train_index(args.repo, args.index, args.nlist, args.m_value, args.nbits,
                args.slide_prop, args.patches_per_slide, args.min_slides_project,
                args.max_slides_project, args.workers, args.token)