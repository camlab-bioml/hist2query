# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-09-23

### Changed

- Move embedding to device casting for GPU inside `asyncio.Lock()`
- Ensure L2 normalization match for embedding addition

## [0.3.0] - 2026-08-24

### Added

- **Draft/Experimental**: Add `chat` endpoint for `Prism2` dialogue 
(only available on GPU): yes/no + open response (both `UNI2` and `Prism2` on GPU)
- Ability to include or exclude projects or tissue from TCGA indexing #4

### Changed

- Make CLI index and metadata optional for `serve`: if not specified, 
`Prism2` is available (only on CUDA) and the `UNI2` model is not loaded
- model inference modes use asyncio locking
- FAISS metadata to `polars` operations

## [0.2.0] - 2026-08-05

### Added

- Support PNG encoding for patch in POST search
- slide and patch parameters for `train` and `add` can accept `None` to use the entire dataset
- LICENSE (MIT)

### Changed

- thread pooling now returns file paths instead of arrays, slide processing handled after result
- Adding writes parquet metadata in per-slide chunks

### Fixed

- Incorrect mapping of BRCA study codes to disease description

## [0.1.0] - 2026-07-17

- Initial dev release with CLI sub-parsers 
for `train`, `add`, and `serve`
