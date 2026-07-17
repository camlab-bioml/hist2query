from hist2query.process.add import add_to_index

def configure_parser(parser):

    parser.add_argument("--repo", dest="repo", default="W8Yi/tcga-wsi-uni2h-features",
        help="UNI model name or path")

    parser.add_argument(
        "--index-in", dest="index_in", required=True,
        help="Input path to the trained index on which to add embeddings.")

    parser.add_argument(
        "--index-out", dest="index_out", required=True,
        help="Path to the output index with embeddings added.")

    parser.add_argument(
        "--metadata-out", dest="metadata_out", required=True,
        help="Path to the output parquet file storing the added index metadata.")

    parser.add_argument(
        "--slide-prop", default=0.7, type=float, dest="slide_prop",
        help="Set a proportion of each project to train on. If None, train on everything (NOT recommended).")

    parser.add_argument(
        "--patches-per-slide", default=500, type=int, dest="patches_per_slide",
        help="Number of patches per slide to sub-sample and train on. If None, train on all patches (NOT recommended).")

    parser.add_argument(
        "--min-slides-project", default=150, type=int, dest="min_slides_project",
        help="Minimum number of slides to include in training per project.")

    parser.add_argument(
        "--max-slides-project", default=500, type=int, dest="max_slides_project",
        help="Maximum number of slides to include in training per project.")

    parser.add_argument(
        "--workers", default=16, type=int, dest="workers",
        help="Number of pool workers to use for multi-threaded hf downloads.")

    parser.add_argument("--hf-token", dest="token", type=str, default=None,
                        help="Hugging Face access token.")

    parser.set_defaults(func=run)


def run(args):

    add_to_index(args.repo, args.index_in, args.index_out, args.metadata_out,
                args.slide_prop, args.patches_per_slide, args.min_slides_project,
                args.max_slides_project, args.workers, args.token)