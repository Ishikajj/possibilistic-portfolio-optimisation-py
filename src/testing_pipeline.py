from pathlib import Path

from pipeline_wrapper import (
    calculate_all_functions_perdatasets,
    call_all_datasets,
)
from data_input import (
    load_excess_returns_from_kenneth_french_path,
    prepare_returns,
)


def test_calculate_all_functions():
    """Minimal test for calculate_all_functions_perdatasets."""

    path_dataset = Path("../datasets/10_Industry_Portfolios_Daily.csv")
    path_rf = Path("../datasets/F-F_Research_Data_Factors_daily.csv")

    excess = load_excess_returns_from_kenneth_french_path(
        portfolios_path=path_dataset,
        risk_free_path=path_rf,
        start_date="2000-01-01",
        end_date="2001-01-01",
    )
    returns = prepare_returns(excess)

    folder = calculate_all_functions_perdatasets(
        dataset=returns,
        dataset_name="test_single_dataset",
        burn_in=50,
        max_models=10,
    )

    print("Single dataset test complete:", folder)


def test_call_all_datasets():
    """Minimal test for call_all_datasets."""

    datasets = [
        Path("../datasets/10_Industry_Portfolios_Daily.csv"),
        (
            Path("../datasets/6_Portfolios_size_btm.csv"),
            "Number of Firms in Portfolios",
        ),
    ]

    folders = call_all_datasets(
        portfolios_paths=datasets,
        start_date="2000-01-01",
        end_date="2001-01-01",
        burn_in=50,
        max_models=10,
    )

    print("Multi-dataset test complete:")
    for f in folders:
        print(f)


if __name__ == "__main__":
    test_call_all_datasets()
