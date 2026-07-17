import uvicorn
from hist2query.app.app import (
    create_app,
    load_uni_model,
    load_index,
    load_metadata)

def configure_parser(parser):

    parser.add_argument(
        "--model",default="hf-hub:MahmoodLab/UNI2-h",
        help="UNI model name or path")

    parser.add_argument("--index", required=True, help="Path to FAISS index")

    parser.add_argument("--metadata", required=True,
        help="Path to metadata parquet file matching the --index.")

    parser.add_argument("--host", default="127.0.0.1")

    parser.add_argument("--port", default=7000, type=int)
    
    parser.add_argument("--workers", default=1, type=int)

    parser.set_defaults(func=run)

def run(args):

    app = create_app(
        model_loader=lambda: load_uni_model(args.model),
        index_loader=lambda: load_index(args.index),
        metadata_loader=lambda: load_metadata(args.metadata))

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)