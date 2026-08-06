# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-06

### Added

- **Draft/Experimental**: Add `chat` endpoint for `Prism2` dialogue (only available on GPU)

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
