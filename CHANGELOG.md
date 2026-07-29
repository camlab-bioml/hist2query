# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-29

### Added

- Support PNG encoding for patch in POST search
- slide and patch parameters for `train` and `add` can accept `None` to use the entire dataset
- LICENSE (MIT)

### Fixed

- Incorrect mapping of BRCA study codes to disease description

## [0.1.0] - 2026-07-17

- Initial dev release with CLI sub-parsers 
for `train`, `add`, and `serve`
