import uvicorn
from hist2query.app.app import (
    create_app,
    load_uni_model,
    load_index,
    load_metadata)

def configure_parser(parser):

    parser.add_argument('-md', "--model", default="hf-hub:MahmoodLab/UNI2-h",
        help="UNI model name or path")

    parser.add_argument('-i', "--index", required=True, help="Path to FAISS index")

    parser.add_argument('-m', "--metadata", required=True,
        help="Path to metadata parquet file matching the --index.")

    parser.add_argument('-hs', "--host", default="127.0.0.1")

    parser.add_argument('-p', "--port", default=7000, type=int)
    
    parser.add_argument('-w', "--workers", default=1, type=int,
        help="Number of workers to use for fastAPI.")

    parser.set_defaults(func=run)

def run(args):

    app = create_app(
        model_loader=lambda: load_uni_model(args.model),
        index_loader=lambda: load_index(args.index),
        metadata_loader=lambda: load_metadata(args.metadata))

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)