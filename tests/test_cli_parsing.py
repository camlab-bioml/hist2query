import argparse
import sys
from unittest.mock import patch
from hist2query.cli.train import run as run_train
from hist2query.cli.add import run as run_add
from hist2query.main import main

@patch("hist2query.cli.train.train_index")
def test_train_cli(mock_train_index):

    args = argparse.Namespace(
        repo="fake/repo", index="test.index",
        nlist=100, m_value=8, nbits=8,
        slide_prop=0.1, patches_per_slide=50,
        min_slides_project=5, max_slides_project=10, workers=2, token=None)

    run_train(args)

    mock_train_index.assert_called_once_with(
        "fake/repo", "test.index", 100, 8, 8, 0.1, 50, 5, 10, 2, None)

@patch("hist2query.cli.train.train_index")
def test_train_cli_none_params(mock_train_index):

    args = argparse.Namespace(
        repo="fake/repo", index="test.index",
        nlist=100, m_value=8, nbits=8,
        slide_prop=None, patches_per_slide=None,
        min_slides_project=None, max_slides_project=None, workers=2, token=None)

    run_train(args)

    mock_train_index.assert_called_once_with(
        "fake/repo", "test.index", 100, 8, 8, None, None, None, None, 2, None)


@patch("hist2query.cli.train.train_index")
def test_train_cli_main(mock_train_index):

    test_args = [
        "hist2query", "train", "--repo", "fake/repo", "--index-out", "test.index",
        "--nlist", "64", "--mquant", "8", "--nbits", "4", "--slide-prop", "0.1",
        "--patches-per-slide", "10", "--min-slides-project", "2",
        "--max-slides-project", "5", "--workers", "2"]

    sys.argv = test_args
    main()
    mock_train_index.assert_called_once_with(
        "fake/repo", "test.index", 64, 8, 4, 0.1, 10, 2, 5, 2, None)

@patch("hist2query.cli.train.train_index")
def test_train_cli_main_none_params(mock_train_index):

    test_args = [
        "hist2query", "train", "--repo", "fake/repo", "--index-out", "test.index",
        "--nlist", "64", "--mquant", "8", "--nbits", "4", "--slide-prop", "None",
        "--patches-per-slide", "none", "--min-slides-project", "none",
        "--max-slides-project", "none", "--workers", "2"]

    sys.argv = test_args
    main()
    mock_train_index.assert_called_once_with(
        "fake/repo", "test.index", 64, 8, 4, None, None, None, None, 2, None)

@patch("hist2query.cli.add.add_to_index")
def test_add_cli(mock_add_index):

    args = argparse.Namespace(
        repo="fake/repo", index_in="test.index",
        index_out = "test_out.index",
        metadata_out = "test.parquet",
        slide_prop=0.1, patches_per_slide=50,
        min_slides_project=5, max_slides_project=10, workers=2, token=None)

    run_add(args)

    mock_add_index.assert_called_once_with(
        "fake/repo", "test.index", "test_out.index",
        "test.parquet", 0.1, 50, 5, 10, 2, None)


@patch("hist2query.cli.add.add_to_index")
def test_add_cli_none_params(mock_add_index):

    args = argparse.Namespace(
        repo="fake/repo", index_in="test.index",
        index_out="test_out.index",
        metadata_out="test.parquet",
        slide_prop=None, patches_per_slide=None,
        min_slides_project=None, max_slides_project=None, workers=2, token=None)

    run_add(args)

    mock_add_index.assert_called_once_with(
        "fake/repo", "test.index", "test_out.index",
        "test.parquet", None, None, None, None, 2, None)

@patch("hist2query.cli.add.add_to_index")
def test_add_cli_main(mock_add_index):

    test_args = [
        "hist2query", "add", "--repo", "fake/repo", "--index-in", "test.index",
        "--index-out", "test_out.index", "--metadata-out", "test.parquet",
        "--slide-prop", "0.1",
        "--patches-per-slide", "10", "--min-slides-project", "2",
        "--max-slides-project", "5", "--workers", "2"]

    sys.argv = test_args
    main()
    mock_add_index.assert_called_once_with(
        "fake/repo", "test.index", "test_out.index",
        "test.parquet", 0.1, 10, 2, 5, 2, None)

@patch("hist2query.cli.add.add_to_index")
def test_add_cli_main_none_params(mock_add_index):

    test_args = [
        "hist2query", "add", "--repo", "fake/repo", "--index-in", "test.index",
        "--index-out", "test_out.index", "--metadata-out", "test.parquet",
        "--slide-prop", "none",
        "--patches-per-slide", "none", "--min-slides-project", "none",
        "--max-slides-project", "none", "--workers", "2"]

    sys.argv = test_args
    main()
    mock_add_index.assert_called_once_with(
        "fake/repo", "test.index", "test_out.index",
        "test.parquet", None, None, None, None, 2, None)