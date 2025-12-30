import pytest
import pandas as pd
import polars as pl
import numpy as np

# Import the module - adjust the import path as needed
from split_dataset import (
    _validate_ratio,
    _calculate_split_ratios,
    _perform_splits,
    _prepare_stratify_df_pd,
    _prepare_stratify_df_pl,
    split_dataset_pd,
    split_dataset_pl,
    split_dataset,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pd_df():
    """Create a sample pandas DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(100),
            "label": [0] * 50 + [1] * 50,
            "value": np.random.randn(100),
        }
    )


@pytest.fixture
def sample_pl_df(sample_pd_df):
    """Create a sample polars DataFrame for testing."""
    return pl.from_pandas(sample_pd_df)


@pytest.fixture
def duplicated_pd_df():
    """Create a pandas DataFrame with duplicated X values."""
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10,
            "label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0] * 10,
            "value": range(100),
        }
    )


@pytest.fixture
def duplicated_pl_df(duplicated_pd_df):
    """Create a polars DataFrame with duplicated X values."""
    return pl.from_pandas(duplicated_pd_df)


@pytest.fixture
def small_pd_df():
    """Create a small pandas DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "id": range(10),
            "label": [0] * 5 + [1] * 5,
        }
    )


@pytest.fixture
def small_pl_df(small_pd_df):
    """Create a small polars DataFrame for edge case testing."""
    return pl.from_pandas(small_pd_df)


# =============================================================================
# Tests for _validate_ratio
# =============================================================================


class TestValidateRatio:
    """Tests for the _validate_ratio function."""

    def test_valid_ratio(self):
        """Test that valid ratios pass without error."""
        _validate_ratio((0.7, 0.2, 0.1))
        _validate_ratio((0.8, 0.2, 0.0))
        _validate_ratio((1.0, 0.0, 0.0))
        _validate_ratio((7, 2, 1))  # Non-normalized ratios

    def test_invalid_length(self):
        """Test that ratios with wrong length raise ValueError."""
        with pytest.raises(ValueError, match="exactly 3 values"):
            _validate_ratio((0.7, 0.3))

        with pytest.raises(ValueError, match="exactly 3 values"):
            _validate_ratio((0.6, 0.2, 0.1, 0.1))

    def test_zero_sum(self):
        """Test that zero sum ratio raises ValueError."""
        with pytest.raises(ValueError, match="Sum of ratios must be positive"):
            _validate_ratio((0.0, 0.0, 0.0))

    def test_negative_ratio(self):
        """Test that negative ratios raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            _validate_ratio((0.7, -0.1, 0.4))

        with pytest.raises(ValueError, match="non-negative"):
            _validate_ratio((-0.5, 0.3, 0.2))


# =============================================================================
# Tests for _calculate_split_ratios
# =============================================================================


class TestCalculateSplitRatios:
    """Tests for the _calculate_split_ratios function."""

    def test_standard_ratio(self):
        """Test standard 70/20/10 split."""
        train_val_ratio, val_test_ratio = _calculate_split_ratios((0.7, 0.2, 0.1))
        assert pytest.approx(train_val_ratio, rel=1e-5) == 0.3
        assert pytest.approx(val_test_ratio, rel=1e-5) == 1 / 3

    def test_no_test_set(self):
        """Test split with no test set."""
        train_val_ratio, val_test_ratio = _calculate_split_ratios((0.7, 0.3, 0.0))
        assert pytest.approx(train_val_ratio, rel=1e-5) == 0.3
        assert pytest.approx(val_test_ratio, rel=1e-5) == 0.0

    def test_no_val_set(self):
        """Test split with no validation set."""
        train_val_ratio, val_test_ratio = _calculate_split_ratios((0.8, 0.0, 0.2))
        assert pytest.approx(train_val_ratio, rel=1e-5) == 0.2
        assert pytest.approx(val_test_ratio, rel=1e-5) == 1.0

    def test_equal_split(self):
        """Test equal three-way split."""
        train_val_ratio, val_test_ratio = _calculate_split_ratios((1, 1, 1))
        assert pytest.approx(train_val_ratio, rel=1e-5) == 2 / 3
        assert pytest.approx(val_test_ratio, rel=1e-5) == 0.5

    def test_only_train(self):
        """Test with only training set."""
        train_val_ratio, val_test_ratio = _calculate_split_ratios((1.0, 0.0, 0.0))
        assert pytest.approx(train_val_ratio, rel=1e-5) == 0.0
        assert pytest.approx(val_test_ratio, rel=1e-5) == 0.0


# =============================================================================
# Tests for _perform_splits
# =============================================================================


class TestPerformSplits:
    """Tests for the _perform_splits function."""

    def test_basic_split(self):
        """Test basic splitting without stratification."""
        x_values = list(range(100))
        train_x, val_x, test_x = _perform_splits(
            x_values, None, 0.3, 0.5, random_state=42
        )

        assert len(train_x) == 70
        assert len(val_x) == 15
        assert len(test_x) == 15

        # Check no overlap
        all_values = set(train_x) | set(val_x) | set(test_x)
        assert len(all_values) == 100

    def test_stratified_split(self):
        """Test stratified splitting."""
        x_values = list(range(100))
        y_values = [0] * 50 + [1] * 50

        train_x, val_x, test_x = _perform_splits(
            x_values, y_values, 0.3, 0.5, random_state=42
        )

        assert len(train_x) + len(val_x) + len(test_x) == 100

    def test_no_test_split(self):
        """Test split with val_test_ratio=0 (no test set)."""
        x_values = list(range(100))
        train_x, val_x, test_x = _perform_splits(
            x_values, None, 0.3, 0.0, random_state=42
        )

        assert len(train_x) == 70
        assert len(val_x) == 30
        assert len(test_x) == 0

    def test_reproducibility(self):
        """Test that same random_state produces same results."""
        x_values = list(range(100))

        result1 = _perform_splits(x_values, None, 0.3, 0.5, random_state=42)
        result2 = _perform_splits(x_values, None, 0.3, 0.5, random_state=42)

        assert result1[0] == result2[0]  # train
        assert result1[1] == result2[1]  # val
        assert result1[2] == result2[2]  # test

    def test_different_random_states(self):
        """Test that different random_states produce different results."""
        x_values = list(range(100))

        result1 = _perform_splits(x_values, None, 0.3, 0.5, random_state=42)
        result2 = _perform_splits(x_values, None, 0.3, 0.5, random_state=123)

        assert result1[0] != result2[0]


# =============================================================================
# Tests for Pandas Implementation
# =============================================================================


class TestPrepareStratifyDfPd:
    """Tests for _prepare_stratify_df_pd function."""

    def test_no_duplicates_with_y(self, sample_pd_df):
        """Test preparation without duplicates, with y."""
        result = _prepare_stratify_df_pd(sample_pd_df, "id", "label", False)

        assert list(result.columns) == ["x", "y"]
        assert len(result) == 100

    def test_no_duplicates_without_y(self, sample_pd_df):
        """Test preparation without duplicates, without y."""
        result = _prepare_stratify_df_pd(sample_pd_df, "id", None, False)

        assert list(result.columns) == ["x"]
        assert len(result) == 100

    def test_with_duplicates_with_y(self, duplicated_pd_df):
        """Test preparation with duplicates, with y."""
        result = _prepare_stratify_df_pd(duplicated_pd_df, "id", "label", True)

        assert list(result.columns) == ["x", "y"]
        assert len(result) == 5  # 5 unique ids

    def test_with_duplicates_without_y(self, duplicated_pd_df):
        """Test preparation with duplicates, without y."""
        result = _prepare_stratify_df_pd(duplicated_pd_df, "id", None, True)

        assert list(result.columns) == ["x"]
        assert len(result) == 5


class TestSplitDatasetPd:
    """Tests for split_dataset_pd function."""

    def test_basic_split(self, sample_pd_df):
        """Test basic train/val split."""
        result = split_dataset_pd(
            sample_pd_df, X="id", train_val_test_ratio=(0.7, 0.3, 0.0)
        )

        assert "dataset" in result.columns
        assert set(result["dataset"].dropna().unique()) == {"train", "val"}

        train_count = (result["dataset"] == "train").sum()
        val_count = (result["dataset"] == "val").sum()

        assert train_count == 70
        assert val_count == 30

    def test_three_way_split(self, sample_pd_df):
        """Test train/val/test split."""
        result = split_dataset_pd(
            sample_pd_df, X="id", train_val_test_ratio=(0.6, 0.2, 0.2)
        )

        assert set(result["dataset"].dropna().unique()) == {"train", "val", "test"}

    def test_stratified_split(self, sample_pd_df):
        """Test stratified splitting maintains class proportions."""
        result = split_dataset_pd(
            sample_pd_df, X="id", y="label", train_val_test_ratio=(0.7, 0.3, 0.0)
        )

        train_data = result[result["dataset"] == "train"]
        val_data = result[result["dataset"] == "val"]

        # Check approximate class balance in both sets
        train_ratio = train_data["label"].mean()
        val_ratio = val_data["label"].mean()

        assert pytest.approx(train_ratio, abs=0.1) == 0.5
        assert pytest.approx(val_ratio, abs=0.1) == 0.5

    def test_custom_names(self, sample_pd_df):
        """Test custom split names."""
        result = split_dataset_pd(
            sample_pd_df,
            X="id",
            train_val_test_name=("training", "validation", "testing"),
        )

        assert "training" in result["dataset"].values
        assert "validation" in result["dataset"].values

    def test_custom_column_name(self, sample_pd_df):
        """Test custom dataset column name."""
        result = split_dataset_pd(sample_pd_df, X="id", dataset_col_name="split_type")

        assert "split_type" in result.columns
        assert "dataset" not in result.columns

    def test_duplicated_x(self, duplicated_pd_df):
        """Test handling of duplicated X values."""
        result = split_dataset_pd(duplicated_pd_df, X="id", X_duplicated=True)

        # All rows with same id should have same dataset
        for id_val in result["id"].unique():
            datasets = result[result["id"] == id_val]["dataset"].unique()
            assert len(datasets) == 1

    def test_reproducibility(self, sample_pd_df):
        """Test that same random_state produces same results."""
        result1 = split_dataset_pd(sample_pd_df, X="id", random_state=42)
        result2 = split_dataset_pd(sample_pd_df, X="id", random_state=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_original_df_unchanged(self, sample_pd_df):
        """Test that original DataFrame is not modified."""
        original_columns = list(sample_pd_df.columns)
        split_dataset_pd(sample_pd_df, X="id")

        assert list(sample_pd_df.columns) == original_columns


# =============================================================================
# Tests for Polars Implementation
# =============================================================================


class TestPrepareStratifyDfPl:
    """Tests for _prepare_stratify_df_pl function."""

    def test_no_duplicates_with_y(self, sample_pl_df):
        """Test preparation without duplicates, with y."""
        result = _prepare_stratify_df_pl(sample_pl_df, "id", "label", False)

        assert result.columns == ["x", "y"]
        assert len(result) == 100

    def test_no_duplicates_without_y(self, sample_pl_df):
        """Test preparation without duplicates, without y."""
        result = _prepare_stratify_df_pl(sample_pl_df, "id", None, False)

        assert result.columns == ["x"]
        assert len(result) == 100

    def test_with_duplicates_with_y(self, duplicated_pl_df):
        """Test preparation with duplicates, with y."""
        result = _prepare_stratify_df_pl(duplicated_pl_df, "id", "label", True)

        assert "x" in result.columns
        assert "y" in result.columns
        assert len(result) == 5

    def test_with_duplicates_without_y(self, duplicated_pl_df):
        """Test preparation with duplicates, without y."""
        result = _prepare_stratify_df_pl(duplicated_pl_df, "id", None, True)

        assert result.columns == ["x"]
        assert len(result) == 5


class TestSplitDatasetPl:
    """Tests for split_dataset_pl function."""

    def test_basic_split(self, sample_pl_df):
        """Test basic train/val split."""
        result = split_dataset_pl(
            sample_pl_df, X="id", train_val_test_ratio=(0.7, 0.3, 0.0)
        )

        assert "dataset" in result.columns
        unique_datasets = set(result["dataset"].drop_nulls().unique().to_list())
        assert unique_datasets == {"train", "val"}

        train_count = result.filter(pl.col("dataset") == "train").height
        val_count = result.filter(pl.col("dataset") == "val").height

        assert train_count == 70
        assert val_count == 30

    def test_three_way_split(self, sample_pl_df):
        """Test train/val/test split."""
        result = split_dataset_pl(
            sample_pl_df, X="id", train_val_test_ratio=(0.6, 0.2, 0.2)
        )

        unique_datasets = set(result["dataset"].drop_nulls().unique().to_list())
        assert unique_datasets == {"train", "val", "test"}

    def test_stratified_split(self, sample_pl_df):
        """Test stratified splitting maintains class proportions."""
        result = split_dataset_pl(
            sample_pl_df, X="id", y="label", train_val_test_ratio=(0.7, 0.3, 0.0)
        )

        train_data = result.filter(pl.col("dataset") == "train")
        val_data = result.filter(pl.col("dataset") == "val")

        train_ratio = train_data["label"].mean()
        val_ratio = val_data["label"].mean()

        assert pytest.approx(train_ratio, abs=0.1) == 0.5
        assert pytest.approx(val_ratio, abs=0.1) == 0.5

    def test_custom_names(self, sample_pl_df):
        """Test custom split names."""
        result = split_dataset_pl(
            sample_pl_df,
            X="id",
            train_val_test_name=("training", "validation", "testing"),
        )

        assert "training" in result["dataset"].to_list()
        assert "validation" in result["dataset"].to_list()

    def test_custom_column_name(self, sample_pl_df):
        """Test custom dataset column name."""
        result = split_dataset_pl(sample_pl_df, X="id", dataset_col_name="split_type")

        assert "split_type" in result.columns
        assert "dataset" not in result.columns

    def test_duplicated_x(self, duplicated_pl_df):
        """Test handling of duplicated X values."""
        result = split_dataset_pl(duplicated_pl_df, X="id", X_duplicated=True)

        # All rows with same id should have same dataset
        for id_val in result["id"].unique().to_list():
            datasets = result.filter(pl.col("id") == id_val)["dataset"].unique()
            assert len(datasets) == 1

    def test_reproducibility(self, sample_pl_df):
        """Test that same random_state produces same results."""
        result1 = split_dataset_pl(sample_pl_df, X="id", random_state=42)
        result2 = split_dataset_pl(sample_pl_df, X="id", random_state=42)

        assert result1.equals(result2)


# =============================================================================
# Tests for Unified Interface
# =============================================================================


class TestSplitDataset:
    """Tests for the unified split_dataset function."""

    def test_pandas_detection(self, sample_pd_df):
        """Test that pandas DataFrame is correctly detected."""
        result = split_dataset(sample_pd_df, X="id")

        assert isinstance(result, pd.DataFrame)
        assert "dataset" in result.columns

    def test_polars_detection(self, sample_pl_df):
        """Test that polars DataFrame is correctly detected."""
        result = split_dataset(sample_pl_df, X="id")

        assert isinstance(result, pl.DataFrame)
        assert "dataset" in result.columns

    def test_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            split_dataset({"id": [1, 2, 3]}, X="id")

        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            split_dataset([[1, 2], [3, 4]], X="id")

    def test_pandas_polars_consistency(self, sample_pd_df, sample_pl_df):
        """Test that pandas and polars produce consistent results."""
        result_pd = split_dataset(sample_pd_df, X="id", random_state=42)
        result_pl = split_dataset(sample_pl_df, X="id", random_state=42)

        # Convert polars to pandas for comparison
        result_pl_pd = result_pl.to_pandas()

        # Compare dataset assignments
        pd.testing.assert_series_equal(
            result_pd["dataset"].reset_index(drop=True),
            result_pl_pd["dataset"].reset_index(drop=True),
            check_names=False,
        )

    def test_all_parameters_passed(self, sample_pd_df):
        """Test that all parameters are correctly passed through."""
        result = split_dataset(
            sample_pd_df,
            X="id",
            y="label",
            train_val_test_ratio=(0.6, 0.2, 0.2),
            X_duplicated=False,
            train_val_test_name=("tr", "va", "te"),
            dataset_col_name="my_split",
            random_state=123,
        )

        assert "my_split" in result.columns
        assert set(result["my_split"].dropna().unique()) == {"tr", "va", "te"}


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_only_train_set(self, sample_pd_df):
        """Test with only training set (val and test = 0)."""
        result = split_dataset_pd(
            sample_pd_df, X="id", train_val_test_ratio=(1.0, 0.0, 0.0)
        )

        # All should be in train
        assert (result["dataset"] == "train").all()

    def test_minimum_samples(self):
        """Test with minimum number of samples."""
        df = pd.DataFrame({"id": [1, 2], "label": [0, 1]})

        result = split_dataset_pd(df, X="id", train_val_test_ratio=(0.5, 0.5, 0.0))

        assert len(result) == 2

    def test_large_dataset(self):
        """Test with a larger dataset."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "id": range(10000),
                "label": np.random.choice([0, 1], 10000),
            }
        )

        result = split_dataset_pd(
            df, X="id", y="label", train_val_test_ratio=(0.7, 0.2, 0.1)
        )

        train_count = (result["dataset"] == "train").sum()
        val_count = (result["dataset"] == "val").sum()
        test_count = (result["dataset"] == "test").sum()

        assert pytest.approx(train_count, rel=0.05) == 7000
        assert pytest.approx(val_count, rel=0.05) == 2000
        assert pytest.approx(test_count, rel=0.05) == 1000

    def test_non_normalized_ratios(self, sample_pd_df):
        """Test with non-normalized ratios."""
        result1 = split_dataset_pd(
            sample_pd_df, X="id", train_val_test_ratio=(70, 30, 0), random_state=42
        )
        result2 = split_dataset_pd(
            sample_pd_df, X="id", train_val_test_ratio=(0.7, 0.3, 0.0), random_state=42
        )

        pd.testing.assert_frame_equal(result1, result2)

    def test_multiclass_stratification(self):
        """Test stratification with multiple classes."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "id": range(300),
                "label": [0] * 100 + [1] * 100 + [2] * 100,
            }
        )

        result = split_dataset_pd(
            df, X="id", y="label", train_val_test_ratio=(0.7, 0.3, 0.0)
        )

        train_data = result[result["dataset"] == "train"]
        val_data = result[result["dataset"] == "val"]

        # Check class distribution in train
        train_dist = train_data["label"].value_counts(normalize=True)
        for cls in [0, 1, 2]:
            assert pytest.approx(train_dist[cls], abs=0.05) == 1 / 3

    def test_string_labels(self):
        """Test with string labels."""
        df = pd.DataFrame(
            {
                "id": range(100),
                "label": ["cat"] * 50 + ["dog"] * 50,
            }
        )

        result = split_dataset_pd(df, X="id", y="label")

        assert "dataset" in result.columns

    def test_string_ids(self):
        """Test with string IDs."""
        df = pd.DataFrame(
            {
                "id": [f"sample_{i}" for i in range(100)],
                "label": [0] * 50 + [1] * 50,
            }
        )

        result = split_dataset_pd(df, X="id", y="label")

        assert "dataset" in result.columns
        assert (result["dataset"].notna()).all()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for real-world usage patterns."""

    def test_full_workflow_pandas(self):
        """Test complete workflow with pandas."""
        # Create dataset
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "patient_id": range(200),
                "diagnosis": np.random.choice(["healthy", "sick"], 200),
                "age": np.random.randint(20, 80, 200),
            }
        )

        # Split dataset
        result = split_dataset(
            df,
            X="patient_id",
            y="diagnosis",
            train_val_test_ratio=(0.6, 0.2, 0.2),
            dataset_col_name="split",
            random_state=42,
        )

        # Verify splits
        train_df = result[result["split"] == "train"]
        val_df = result[result["split"] == "val"]
        test_df = result[result["split"] == "test"]

        assert len(train_df) + len(val_df) + len(test_df) == 200

        # No overlap in patient IDs
        assert len(set(train_df["patient_id"]) & set(val_df["patient_id"])) == 0
        assert len(set(train_df["patient_id"]) & set(test_df["patient_id"])) == 0
        assert len(set(val_df["patient_id"]) & set(test_df["patient_id"])) == 0

    def test_full_workflow_polars(self):
        """Test complete workflow with polars."""
        # Create dataset
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "patient_id": range(200),
                "diagnosis": np.random.choice(["healthy", "sick"], 200),
                "age": np.random.randint(20, 80, 200),
            }
        )

        # Split dataset
        result = split_dataset(
            df,
            X="patient_id",
            y="diagnosis",
            train_val_test_ratio=(0.6, 0.2, 0.2),
            dataset_col_name="split",
            random_state=42,
        )

        # Verify splits
        train_df = result.filter(pl.col("split") == "train")
        val_df = result.filter(pl.col("split") == "val")
        test_df = result.filter(pl.col("split") == "test")

        assert train_df.height + val_df.height + test_df.height == 200

    def test_repeated_measurements_scenario(self):
        """Test scenario with repeated measurements per subject."""
        # Each patient has multiple measurements (8 patients, 4 per class)
        df = pd.DataFrame(
            {
                "patient_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8] * 5,
                "measurement": list(range(80)),
                "outcome": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] * 5,
            }
        )

        result = split_dataset(
            df,
            X="patient_id",
            y="outcome",
            X_duplicated=True,
            train_val_test_ratio=(0.5, 0.25, 0.25),
        )

        # All measurements from same patient should be in same split
        for patient in result["patient_id"].unique():
            patient_data = result[result["patient_id"] == patient]
            assert patient_data["dataset"].nunique() == 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
