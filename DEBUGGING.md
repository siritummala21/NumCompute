# Debugging log

How I found and fixed bugs, and which test now checks each fix.

## How I worked

1. Reproduce the problem with the smallest input I could find (a few rows,
   one chunk, one extreme value).
2. Write a test that fails and checks the exact right answer. Where the
   answer is not obvious I compare with something independent: NumPy on the
   whole array, the slow loop version, the group's batch function, or a
   value worked out by hand. The 2.1 feedback showed that a test which only
   checks "finite" or "no crash" can pass while the result is wrong.
3. Fix the code and run the whole suite again, so one fix cannot break
   another module.
4. Keep the test, and write the bug down here.

After the 2.1 feedback I also tried every new class with the kinds of input
the marker used: tiny values (1e-16), values around 1e9, NaN and inf, bool or
wrong-type parameters, empty chunks, single samples, and outside components
that only have `fit(X)`.

## Group package numcompute (2.1 feedback)

| # | Feedback | Cause | Fix | Test |
|---|---|---|---|---|
| 1 | `StandardScaler` treated `[1e-16, 2e-16, 3e-16]` as constant | `std < 10*eps` is an absolute threshold, so a tiny but varying column looked constant | A column is constant only if max == min; the std is computed on values scaled by a power of two | `TestScalerTinyAndHugeValues` |
| 2 | Distance between 1e9 and 1e9 + 1 came out as 0 | `|a|^2 + |b|^2 - 2a.b` loses all digits when the points are large and close | Compute the differences directly (in blocks, to keep memory small) | `TestDistancesWithLargeCoordinates` |
| 3 | `grad` accepted `h=nan` | `h <= 0` is False for NaN, so the check let it through | `_check_positive_finite` for `h` and `alpha0`; `tol` and `max_iter` checked too | `TestOptimiserStepValidation` |
| 4 | A transformer with only `fit(X)` failed in `Pipeline`, `FeatureUnion` and `ColumnTransformer` | `fit(X, y)` was hard-coded, and the code chained `fit(...).transform(...)` | `_fit` and `_fit_transform` call `fit` with or without `y` and never use its return value | `TestFitOnlyTransformerCompatibility` |
| 5 | The large-coordinate distance test only checked "finite and >= 0" | The test did not state the expected value | It now checks `[[0, 1], [1, 0]]` exactly | `test_pairwise_self_distance_zero_and_errors` |

All 99 original group tests still pass after these changes.

## numcompute_stream (2.2)

| # | What went wrong | Cause | Fix | Test |
|---|---|---|---|---|
| 6 | `import numcompute_stream` failed with "Cannot create a consistent method resolution order" | `StandardScaler(BaseEstimator, TransformerMixin, group.StandardScaler)` listed `BaseEstimator` before a class that already inherits from it | Inherit from `TransformerMixin` and the group class only | every preprocessing test |
| 7 | `gini([0, 0])` returned 1, not 0 | `1 - sum(p^2) * mask` multiplied only the sum by the empty-node mask | `(1 - sum(p^2)) * mask` | `test_gini_and_entropy_values` |
| 8 | `predict` before the first retrain raised `TypeError: 'NoneType' object is not subscriptable` instead of `NotFittedError` | `self._tree["counts"][self.apply(X)]` reads `self._tree` before `apply` gets to check it | Call `apply` first | `test_buffer_retrains_only_when_full` |
| 9 | A chunk with fewer columns was accepted right after a retrain | The width check compared against the buffer, which is empty after a retrain | Compare against `n_features_in_` when the buffer is empty | `test_wider_chunk_pads_older_rows` |
| 10 | A new site from the one-hot encoder crashed `np.concatenate` in the buffer | Older buffered chunks had one column fewer | New categories are added as the last column, and older buffered rows are padded with 0 | `test_new_category_mid_stream_does_not_break_the_tree` |
| 11 | `RuntimeWarning` from the split search on columns with NaN | Right-child counts went negative in the NaN part of the sorted column | Clamp at 0; those positions are not valid splits anyway | `test_vectorised_split_matches_loop` |
| 12 | DDM said "drift" on the first mistake of a perfect model | With no errors `p_min = s_min = 0`, so any error is above `p_min + 3 s_min` | Laplace smoothing: `p = (errors + 1) / (n + 2)` | `test_ddm_first_error_is_not_drift` |
| 13 | The drift at chunk 30 was never detected | I reset DDM at every scheduled retrain, so it learned its reference level on the drifted data | Reset DDM only after a detected drift | `test_drift_triggers_an_early_retrain` |
| 14 | Retraining after drift did not help much | `flush()` retrained mostly on rows from before the drift | `flush(last_n=...)` keeps only the rows since the warning started | `test_flush_keeps_only_last_rows` |
| 15 | Mixed number and string labels gave a wrong confusion matrix | `searchsorted` across dtypes | `TypeError` when the label type changes | `test_confusion_matrix_label_type_change_is_rejected` |
| 16 | `iter_csv_chunks("missing.csv", 10)` only failed at the first `next()` | The checks were inside the generator body | Check first, then return an inner generator | `test_iter_csv_chunks_checks_arguments_straight_away` |
| 17 | A CSV saved from Excel failed with `KeyError: '\ufefftemperature'` | The byte-order mark stuck to the first column name | Open with `utf-8-sig` | `test_excel_bom_and_blank_lines` |
| 18 | One empty chunk stopped the whole stream | `partial_fit` rejected 0 rows | Empty chunks are ignored everywhere | `test_empty_chunk_is_skipped_and_max_chunks` |
| 19 | My first imputer test expected the median of `[1, 1, 4]` to be 2 | Mistake in the test, not the code | Fixed the test data and worked the expected values out by hand | `test_imputer_strategies` |

### Probing after the first full version

Once everything worked I probed the streaming classes the same way the 2.1
marker probed the group code: extreme values, outside components, and a
`warnings`-as-errors run so silent overflows show up.

| # | What went wrong | Cause | Fix | Test |
|---|---|---|---|---|
| 20 | The streaming `StandardScaler` turned `[-1e308, 0, 1e308]` into ±0.556 instead of ±1.225 (the group scaler was already right) | `(x - mean)^2` overflows to inf inside the running variance | `RunningStats` divides each feature by a power of two before squaring; that division is exact | `test_huge_values_are_scaled_correctly`, `test_huge_values_do_not_overflow` |
| 21 | `MinMaxScaler` gave NaN for the same column | `hi - lo` overflowed | Work with half values: `hi/2 - lo/2` cannot overflow | `test_huge_values_are_scaled_correctly` |
| 22 | Histogram edges were NaN for a range of ±1e308 (and the group `histogram` crashed with an index error on it) | `linspace` and the bin formula compute `hi - lo` | Build the edges and count on half values, which gives the same bins | `test_huge_values_do_not_overflow` |
| 23 | Overflow warning when the tree split between -1e308 and 1e308 | The midpoint `lo + (hi - lo) / 2` | Use `lo/2 + hi/2` when `hi - lo` is not finite | `test_split_between_huge_values` |
| 24 | A step with only `transform()` was rejected by `Pipeline`, although my docstring said it was allowed | The group `Pipeline` checks for both `fit` and `transform` | Own `_validate`; `fit` applies such steps without fitting them | `test_transform_only_step_is_allowed` |

The benchmark tables now also start with the run time and the git commit of
the code they measured, as the 2.1 feedback asked for.

## Edge cases I tested on purpose

NaN features and all-NaN columns, NaN and inf labels, constant columns,
empty chunks, 1-D input, Excel CSV files, ties between thresholds and
features, a single class, new classes and new categories in the middle of
the stream, wider and narrower chunks, a label type change, prediction before
the first retrain, bad hyper-parameters (negative, bool, float for int),
values near ±1e308 and 1e-16, and outside components with `fit(X)` only or
`transform` only.
