
# hist2query request examples

Examples of POST requests for `hist2query` search. 

## queryPNG.js

Example of a NodeJS query on a kidney H&E patch. 

To run, first install `hist2query` and execute `serve` using
the test fixture index and metadata:

```commandline
cd hist2query
conda activate hist2query
hist2query serve -i tests/fixtures/test_index_added.index -m tests/fixtures/test_metadata.parquet -p 7000
```

### Querying TCGA slides with UNI2

In another terminal session, install and execute the example script:

```commandline
cd hist2query/examples/
npm run install
node queryTCGAUNI2.js
```

The query output is printed to console, and should look something like:

```commandline
{
  hits: [
    {
      project: 'TCGA-KICH',
      tissue: 'Kidney Chromophobe',
      slide: 'TCGA-KO-8416-01Z-00-DX1.h5',
      x: 71168,
      y: 16384,
      similarity: 0.40631651878356934
    },
    {
      project: 'TCGA-KICH',
      tissue: 'Kidney Chromophobe',
      slide: 'TCGA-KO-8416-01Z-00-DX1.h5',
      x: 73216,
      y: 13312,
      similarity: 0.3673453629016876
    },
    {
      project: 'TCGA-KICH',
      tissue: 'Kidney Chromophobe',
      slide: 'TCGA-KL-8331-01Z-00-DX1.h5',
      x: 16384,
      y: 24576,
      similarity: 0.3433022201061249
    },
    {
      project: 'TCGA-KIRC',
      tissue: 'Kidney renal clear cell carcinoma',
      slide: 'TCGA-CJ-4887-01Z-00-DX1.h5',
      x: 10752,
      y: 32768,
      similarity: 0.3372335135936737
    },
    {
      project: 'TCGA-KIRC',
      tissue: 'Kidney renal clear cell carcinoma',
      slide: 'TCGA-CJ-4887-01Z-00-DX1.h5',
      x: 11776,
      y: 35840,
      similarity: 0.30886775255203247
    }
  ],
  url: {
    'TCGA-KL-8331-01Z-00-DX1.h5': 'https://portal.gdc.cancer.gov/files/ec59c896-8f66-451b-838a-e12f26861a3c',
    'TCGA-KO-8416-01Z-00-DX1.h5': 'https://portal.gdc.cancer.gov/files/6ee84288-0e2a-4cb3-8bda-54cfc162a9c8',
    'TCGA-CJ-4887-01Z-00-DX1.h5': 'https://portal.gdc.cancer.gov/files/2678f782-73c0-4cb0-8686-270c3fa93f97'
  }
}
```

### Querying Prism2 with chat functionality

```commandline
node queryPrism2Chat.js
```

**NOTE**: GPU and CUDA must be available on the hist2query deployment, otherwise you'll receive a 503 error:

```commandline
    data: {
      detail: 'Prism2 not available: CUDA not found in the hist2query deployment.'
    }
  },
  status: 503
}
```


