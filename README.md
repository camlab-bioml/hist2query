# hist2query

Query any H&E patch against TCGA `UNI2` embeddings using FAISS, or with dialogue using `Prism2`.

`hist2query` is comprised of the following CLI functions:

- `train`: Generate a new [TCGA UNI](https://huggingface.co/datasets/W8Yi/tcga-wsi-uni2h-features) [faiss](https://github.com/facebookresearch/faiss)
index using subsampling of the entire dataset
- `add`: Embed embeddings from the same dataset into a trained index for fast lookup with matched metadata in parquet format
- `serve`: deploy a `fastAPI` instance to serve query results from any application

## Getting started

**NOTE**: The [UNI2](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2](https://huggingface.co/paige-ai/Virchow2), 
and [Prism2](https://huggingface.co/paige-ai/Prism2) foundation model hosted on huggingface are gated, so users will need to have a registered account and 
accept the terms of use before installation. **By default, hist2query will install and run the UNI2 model, but not 
the Virchow2 or Prism2 models, as they require GPU access (see below)**. Once accepted, users should generate and set the `HF_TOKEN` env variable before
installation. 

```commandline
export HF_TOKEN="your_hf_token"
```

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

To use the `Virchow2` and `Prism2` models for chat dialogue with the `chat` endpoint:

```commandline
pip install -e .["prism2"]
```

**NOTE**: some of the `Prism2` dependencies such as `torch` and `flash_attn` may conflict 
with the dependency versions for `UNI2`. It is recommended to run `hist2query
through Docker to access `Prism2` (see [Docker](#docker)). 

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

#### Python

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
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"k": 1, "url": True}, timeout=300)
        response.raise_for_status()
        print(response.json())
```

```
{'hits': [
    {'project': 'TCGA-KICH', 'tissue': 'Kidney Chromophobe', slide': 'TCGA-KN-8422-01Z-00-DX1.h5', 'x': 36864, 'y': 8704, 'similarity': 0.4805717468261719}],
 'url': 
    {'TCGA-KN-8422-01Z-00-DX1.h5': 'https://portal.gdc.cancer.gov/files/92518281-c255-4353-b349-715d9e0a936f'}}
```

#### NodeJS or browser

PNG encoding is the recommended method for encoding patches from any JS-derived framework:

```commandline
const pngBuffer = img.png().toBuffer();
const form = new FormData();

    form.append(
        "patch",
        pngBuffer,
        {
            filename: "patch.png",
            contentType: "image/png"
        }
    );
    
const response = await axios.post(
        "http://localhost:7000/search",
        form,
        {
            headers: form.getHeaders(),
            timeout: 300000
        }
    );
```

A full example of a request made through Node can be found in the [examples directory](./examples/queryPNG.js)

The response will contain two fields, `hits` and `url`. 
`hits` will provide a list of query results per H&E patch containing 
project, [tissue](https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations), slide, spatial (coordinate), and query similarity information. 
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

### Chat with `Prism2`: (**NOTE: Experimental endpoint**)

**IMPORTANT**: Using the `Prism2` chat endpoint requires CUDA/GPU. It is recommended to run from docker, 
and/or to install the optional dependencies. By default, `hist2query` only installs the requirements for `UNI2`. 

```commandline
pip install -e .["prism2"]
```

```commandline
response = requests.post(f"http://localhost:7000}/chat",
                                 files={"patch": ("patch.npz", serialize_crop(crop.astype(np.uint8)))},
                                 data={"question": "What type of tissue is this?"}, timeout=300)
        response.raise_for_status()
        print(response.json())
```


## Docker

The `hist2query` fastAPI server can be run through Docker. To enable the container
to access the gated models, deployment requires either:

- passing a hf token as an environment variable `HF_TOKEN` (**RECOMMENDED**):
```commandline
export HF_TOKEN="your_hf_token"
docker run -p 7000:7000 -v /home/:/home/ -e HF_TOKEN=$HF_TOKEN hist2query:latest hist2query serve -hs 0.0.0.0
```

- mounting a local hf cache that contains the models, pre-downloaded:
```commandline
docker run -p 7000:7000 -v /home/:/home/ -v ~/.cache/huggingface:/root/.cache/huggingface hist2query:latest hist2query serve -hs 0.0.0.0
```

**Importantly**, the `-hs` host option should be set to `0.0.0.0`
for access outside the container. Currently, the container exposes two ports, 6000 and 7000. 