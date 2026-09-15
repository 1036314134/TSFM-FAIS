"""Check the two-output imputation counterexample using exact integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path


def integrate(coefficients, lower, upper):
    """Integrate a polynomial specified in ascending powers."""
    return sum(
        value * (upper ** (index + 1) - lower ** (index + 1)) / (index + 1)
        for index, value in enumerate(coefficients)
    )


def squared_error(point):
    a, b = map(Q, point)
    # Uniform density 1/2 and mean over the two forecast coordinates 1/2.
    return integrate([a * a + b * b, -2 * a, 1 - 2 * b, Q(0), Q(1)], Q(-1), Q(1)) / 4


def absolute_error(point, second_root):
    a, b = map(Q, point)
    root = Q(second_root)
    if root < 0 or root * root != b:
        raise ValueError("the supplied root must be an exact nonnegative square root")
    cuts = sorted({Q(-1), Q(1), *(x for x in (a, -root, root) if -1 < x < 1)})
    result = Q(0)
    for lower, upper in zip(cuts[:-1], cuts[1:], strict=True):
        middle = (lower + upper) / 2
        sign1 = 1 if middle >= a else -1
        sign2 = 1 if middle * middle >= b else -1
        result += integrate([-sign1 * a - sign2 * b, Q(sign1), Q(sign2)], lower, upper) / 4
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed checks")
    output.mkdir(parents=True, exist_ok=True)

    mse_minimum = Q(4, 15)
    mae_minimum = Q(79, 192)
    records = []
    # Exact piecewise integration is independent of the closed-form risk formulas.
    for a in [Q(-3), Q(-1), Q(-3, 4), Q(-1, 4), Q(0), Q(1, 4), Q(1, 3), Q(1), Q(3)]:
        r = abs(a)
        mse = squared_error((a, a * a))
        mae = absolute_error((a, a * a), r)
        assert mse == mse_minimum + a**2 / 6 + a**4 / 2
        if r <= 1:
            # The factorization proves the global minimum within [-1, 1].
            assert mae == mae_minimum + (r - Q(1, 4)) ** 2 * (2 * r / 3 + Q(1, 12))
        else:
            assert mae == (r + r * r - Q(1, 3)) / 2
            assert mae > Q(5, 6) > mae_minimum
        records.append({"imputation": str(a), "mae": str(mae), "mse": str(mse)})

    mean_mse = squared_error((Q(0), Q(1, 3)))
    median_mse = squared_error((Q(0), Q(1, 4)))
    median_mae = absolute_error((Q(0), Q(1, 4)), Q(1, 2))
    assert mean_mse == Q(19, 90)
    assert median_mse == Q(103, 480) < mse_minimum
    assert median_mae == Q(3, 8) < mae_minimum
    assert mse_minimum - mean_mse == Q(1, 18)
    assert mae_minimum - median_mae == Q(7, 192)
    assert mean_mse * Q(5, 4) == Q(19, 72) < mse_minimum

    # Check the irrational mean-forecast MAE by deterministic numerical integration.
    count = 100_000
    numerical_mae = (
        sum(
            (abs(z) + abs(z * z - 1 / 3)) / 2
            for z in (-1 + (2 * index + 1) / count for index in range(count))
        )
        / count
    )
    mean_mae = Q(1, 4) + 2 / (9 * 3**0.5)
    assert abs(numerical_mae - mean_mae) < 1e-8
    assert mean_mae < float(mae_minimum)
    manifest = {
        "status": "completed",
        "role": "analytic counterexample; no empirical TSFM or novelty claim",
        "data": "Z uniform on [-1,1]; Y=(Z,Z^2); observed context fixed",
        "fixed_forecaster": "F(a)=(a,a^2); any real-valued single imputation allowed",
        "metric": "mean over two output coordinates in the specified example units",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "single_imputation_optimum": {"mae": str(mae_minimum), "mse": str(mse_minimum)},
        "predictive_mean": {"mae": float(mean_mae), "mse": str(mean_mse)},
        "predictive_coordinate_median": {"mae": str(median_mae), "mse": str(median_mse)},
        "exact_integration_cases": records,
        "numerical_mean_mae": numerical_mae,
        "four_independent_predictive_draws_mean_expected_mse": "19/72",
        "assumptions": [
            "imputation randomness is independent of the unobserved response conditional on observations",
            "posterior distribution is known only in this constructed example",
            "the real TSFM outputs need not be complete-data conditional means",
            "empirical candidate forecasts are not established posterior samples",
        ],
    }
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
