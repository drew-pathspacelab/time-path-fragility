from src.data.interface import get_returns
from src.model.regression import rolling_forward_decompose, sample_forward_decompose
from src.model.risk import rolling_risk_decomposition, sample_risk_decomposition
from src.data.universe import load_universe


TEST_UNIVERSE = load_universe()

ASSETS = TEST_UNIVERSE.assets
FACTORS = TEST_UNIVERSE.market

START = "2020-01-01"
END = "2026-01-01"
WINDOW = 252


def main():

    asset_returns = get_returns(ASSETS, START, END)
    factor_returns = get_returns(FACTORS, START, END, is_factor=True)

    rolling_model_ds = rolling_forward_decompose(asset_da=asset_returns,
                                                 factor_da=factor_returns,
                                                 window=WINDOW,
                                                 include_intercept=True,
                                                 )

    sample_model_ds = sample_forward_decompose(asset_returns,
                                               factor_returns,
                                               include_intercept=True
                                               )

    rolling_risk = rolling_risk_decomposition(rolling_model_ds, factor_returns)
    sample_risk = sample_risk_decomposition(sample_model_ds, factor_returns)

    print("Pipeline complete.")
    print(rolling_risk)
    print(sample_risk)


if __name__ == "__main__":
    main()
