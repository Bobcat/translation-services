"""Fixture-location resolution: unique-stem identity with source-mirrored nesting under _regression.

The workbench only knows an image's bare filename stem, so the stem is the identifier; fixtures are
nested to mirror the source image's subdir (``testset/docpack/07_…`` -> ``_regression/docpack/07_…``)
and every endpoint resolves stem -> nested path.
"""
import json
from pathlib import Path

import pytest

from app.regression import capture as C


def _img(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def _fixture(reg: Path, rel_name: str, lang: str = "nl", variant: str = "v1") -> Path:
    d = reg / rel_name / lang / variant
    d.mkdir(parents=True, exist_ok=True)
    (d / "fixture.json").write_text(
        json.dumps({"target_lang": lang, "hint_translations": {}, "leftover_translations": {}})
    )
    (d / "snapshot.json").write_text(json.dumps({"reocr": []}))
    return d


def test_testset_image_finds_subdir_source_and_prunes_underscore_dirs(tmp_path):
    _img(tmp_path, "docpack/07_x.png")
    assert C.testset_image("07_x", testset_root=tmp_path) == tmp_path / "docpack" / "07_x.png"
    _img(tmp_path, "_regression/07_x/nl/v1/source.png")  # a fixture copy, not a source
    assert C.testset_image("source", testset_root=tmp_path) is None


def test_source_reldir_root_subdir_and_ambiguous(tmp_path):
    _img(tmp_path, "kassabon.jpg")
    _img(tmp_path, "docpack/07_x.png")
    assert C.source_reldir("kassabon", testset_root=tmp_path) == ""
    assert C.source_reldir("07_x", testset_root=tmp_path) == "docpack"
    assert C.source_reldir("missing", testset_root=tmp_path) == ""  # fresh upload -> root
    _img(tmp_path, "07_x.png")  # same stem now flat AND under docpack -> invariant violated
    with pytest.raises(ValueError):
        C.source_reldir("07_x", testset_root=tmp_path)


def test_resolve_fixture_root_mirrors_source(tmp_path):
    reg, ts = tmp_path / "_regression", tmp_path / "testset"
    _img(ts, "docpack/07_x.png")
    _fixture(reg, "docpack/07_x")
    assert C.resolve_fixture_root("07_x", regression_root=reg, testset_root=ts) == reg / "docpack" / "07_x"


def test_resolve_fixture_root_falls_back_to_tree_walk_when_source_gone(tmp_path):
    reg, ts = tmp_path / "_regression", tmp_path / "testset"
    ts.mkdir()
    _fixture(reg, "annual/unicef/13_x")  # nested two deep, no source image on disk
    assert C.resolve_fixture_root("13_x", regression_root=reg, testset_root=ts) == reg / "annual" / "unicef" / "13_x"


def test_list_fixtures_reports_nested_reldir(tmp_path):
    reg, ts = tmp_path / "_regression", tmp_path / "testset"
    _img(ts, "docpack/07_x.png")
    _img(ts, "kassabon.jpg")
    _fixture(reg, "docpack/07_x")
    _fixture(reg, "kassabon")
    out = {i["name"]: i for i in C.list_fixtures(regression_root=reg, testset_root=ts)}
    assert out["07_x"]["reldir"] == "docpack" and out["07_x"]["in_testset"] is True
    assert out["kassabon"]["reldir"] == "" and out["kassabon"]["in_testset"] is True
    assert set(out["07_x"]["langs"]) == {"nl"}


def test_list_subdirs_returns_nested_non_underscore_dirs(tmp_path):
    _img(tmp_path, "docpack/07_x.png")
    _img(tmp_path, "annual/unicef/13_x.png")
    _img(tmp_path, "kassabon.jpg")            # a root image contributes no subdir
    (tmp_path / "_regression" / "07_x" / "nl" / "v1").mkdir(parents=True)  # pruned
    assert C.list_subdirs(testset_root=tmp_path) == ["annual", "annual/unicef", "docpack"]


def test_delete_path_finds_nested_fixture_by_bare_stem(tmp_path):
    reg = tmp_path / "_regression"
    _fixture(reg, "docpack/07_x")
    assert C.delete_path("07_x", regression_root=reg) is True
    assert not (reg / "docpack" / "07_x").exists()
