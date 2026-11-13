import numpy as np
import pytest

from numcompute_stream import io as nio


def test_read_csv_splits_numeric_categorical_and_target(csv_file):
    ds = nio.read_csv(csv_file, target="label", categorical=["c"])
    assert ds.numeric_names == ["a", "b"]
    assert ds.categorical_names == ["c"]
    assert ds.X.shape == (4, 2) and ds.X_cat.shape == (4, 1)
    assert ds.y.dtype.kind == "i"
    np.testing.assert_array_equal(ds.y, [0, 1, 1, 0])


def test_missing_tokens_become_nan_and_none(csv_file):
    ds = nio.read_csv(csv_file, target="label", categorical=["c"])
    assert np.isnan(ds.X[2, 0])          # "NA"
    assert np.isnan(ds.X[1, 1])          # empty field
    assert ds.X_cat[3, 0] is None


def test_chunks_together_equal_the_whole_file(csv_file):
    whole = nio.read_csv(csv_file, target="label", categorical=["c"])
    chunks = list(nio.iter_csv_chunks(csv_file, 3, target="label", categorical=["c"]))
    assert [len(c) for c in chunks] == [3, 1]          # last chunk is smaller
    np.testing.assert_array_equal(np.vstack([c.X for c in chunks]), whole.X)
    np.testing.assert_array_equal(np.concatenate([c.y for c in chunks]), whole.y)


def test_iter_csv_chunks_checks_arguments_straight_away(tmp_path):
    # the errors must come when the function is called, not at the first next()
    with pytest.raises(FileNotFoundError):
        nio.iter_csv_chunks(tmp_path / "nope.csv", 10)
    with pytest.raises(ValueError):
        nio.iter_csv_chunks(tmp_path / "nope.csv", 0)


def test_bad_number_names_the_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,x\n2,y\n")
    with pytest.raises(ValueError, match="'b'"):
        nio.read_csv(p)


def test_row_with_wrong_number_of_fields(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="Line 3"):
        nio.read_csv(p)


def test_excel_bom_and_blank_lines(tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes("\ufeffa,label\n1,0\n\n2,1\n".encode("utf-8"))
    ds = nio.read_csv(p, target="label")
    assert ds.numeric_names == ["a"]
    np.testing.assert_array_equal(ds.X.ravel(), [1.0, 2.0])


def test_string_labels_and_unknown_column(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("a,label\n1,cat\n2,dog\n")
    assert list(nio.read_csv(p, target="label").y) == ["cat", "dog"]
    with pytest.raises(KeyError):
        nio.read_csv(p, target="nope")


def test_missing_target_is_an_error(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,label\n1,0\n2,\n")
    with pytest.raises(ValueError, match="target"):
        nio.read_csv(p, target="label")


def test_chunk_stream_keeps_order_and_shuffle_is_a_permutation(rng):
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    chunks = list(nio.chunk_stream(X, y, chunk_size=4))
    assert [len(c[1]) for c in chunks] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), y)
    shuffled = np.concatenate([c[1] for c in nio.chunk_stream(X, y, 3, shuffle=True,
                                                               random_state=1)])
    assert sorted(shuffled) == list(y) and not np.array_equal(shuffled, y)


def test_write_then_read_round_trip(tmp_path):
    X = np.array([[1.5, "a"], [np.nan, None]], dtype=object)
    p = tmp_path / "out.csv"
    nio.write_csv(p, X, ["num", "cat"], y=[0, 1], target_name="label")
    ds = nio.read_csv(p, target="label", categorical=["cat"])
    assert ds.X[0, 0] == 1.5 and np.isnan(ds.X[1, 0])
    assert ds.X_cat[0, 0] == "a" and ds.X_cat[1, 0] is None
