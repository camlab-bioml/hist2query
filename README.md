# hist2query

Query any H&E patch against TCGA UNI2 embeddings using FAISS. 


`hist2query` is comprised of the following CLI functions:

- `train`: Generate a new [TCGA UNI](https://huggingface.co/datasets/W8Yi/tcga-wsi-uni2h-features) [faiss](https://github.com/facebookresearch/faiss)
index using subsampling of the entire dataset
- `add`: Embed embeddings from the same dataset into a trained index for fast lookup with matched metadata in parquet format
- `serve`: deploy a `fastAPI` instance to serve query results from any application


## Installation

```commandline
conda create --name hist2query python=3.11
conda activate hist2query
cd hist2query
pip install .
```

or for development:

```commandline
pip install -e .["dev"]
```

## Usage

`hist2query -h`

```
usage: hist2query [-h] [-v] {serve,train,add} ...

positional arguments:
  {serve,train,add,inspect}
    serve               Run the FastAPI server
    train               Build and train an initial FAISS index.
    add                 Add embeddings to a trained index.

options:
  -h, --help            show this help message and exit
  -v, --version         Show the current hist2query version then exit. Does not execute the application.
```

### Querying a patch using the `/search` endpoint

```commandline
def serialize_crop(crop: Union[np.array, np.ndarray, None]=None):
    """
    Serialize the WSI crop into compressed bytes for a POST request
    """
    if crop is not None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=np.stack(crop))
        return buffer.getvalue()
    return None
    
crop = np.ones((224, 224, 3))
    
response = requests.post(f"http://localhost:7000}/search",
                                 files={"patch": ("patch.npy", serialize_crop(crop.astype(np.uint8)))},
                                 data={"k": 1, "url": True}, timeout=300)
        response.raise_for_status()
        print(response.json())
```

```
{'hits': [
    {'project': 'TCGA-KICH', 'slide': 'TCGA-KN-8422-01Z-00-DX1.h5', 'x': 36864, 'y': 8704, 'similarity': 0.4805717468261719}],
 'url': 
    {'TCGA-KN-8422-01Z-00-DX1.h5': 'https://portal.gdc.cancer.gov/files/92518281-c255-4353-b349-715d9e0a936f'}}
```

The response will contain two fields, `hits` and `url`. 
`hits` will provide a list of query results per H&E patch containing 
project, slide, spatial (coordinate), and query similarity information. 
`url` will either be `None` if the user doesn't request URLs, or a
set of key value pairs matching every `slide` identifier in the query
list to a URL in GDC portal where the SVS slide can be viewed 
([Example](https://portal.gdc.cancer.gov/files/92518281-c255-4353-b349-715d9e0a936f))

The request can be processed into tabular form for viewing and ranking:

```commandline
frame_results = pd.DataFrame(resp['hits'])
if resp['url']:
    frame_results['url'] = resp['slide'].map(resp['url'])
```

