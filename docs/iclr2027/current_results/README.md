# Supplementary results

These files accompany *Peer Repair and Forecast Fusion for Missing Sensor Histories*.

`all_panel_results.csv` contains the complete fixed snapshot of 1,275 panel--method results. `all_station_results.csv` contains the corresponding site-level scores. `primary_comparisons.csv` contains paired comparisons of the primary procedure with the displayed Beijing controls and leave-one-station-out sensitivity ranges.

The primary identifier is `half_var_long_repair_peer_ridge_peer`. It always denotes peer-regression target repair in the latest 192 hours, long-history Chronos-2 prediction with local and peer variables, and equal fusion with the fitted VAR. This exact primary was evaluated on the three Beijing panels. HDB Gaussian rows are component-transfer controls, not substitute results for that primary.

MAE and MSE use original-observation prefix standard deviations and score only originally observed future values. Windows are averaged within sites and sites are equally weighted. Lower scores are better. Relative percentage change is `100 * (method / reference - 1)`. Leave-one-site-out ranges describe sensitivity to the included sites and are not confidence intervals.

Panel names encode the source, missingness condition and horizon. `legacy_native_h96` is the identifier of the original-missing Beijing horizon-96 cohort. Long inputs have 8,192 hours for Beijing and 349--586 available hours for HDB. `targets` denotes the variables supplied directly to the forecaster; a repair may still use auxiliary observations. `half_var_` denotes equal forecast fusion with the same statistical branch.

The results are developmental. The method and application focus were selected using the observed results; they do not form an independent confirmation study. Per-panel minima do not define a deployable selector. The reserved HDB periods remain unscored.

In the repository, `python scripts/build_current_manuscript.py` rebuilds the four table fragments and the component figure from the fixed score snapshot. The LaTeX source package already includes those generated assets and can be compiled without running this script or accessing the raw datasets.
