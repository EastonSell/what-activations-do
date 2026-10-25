> Public sanitized derivative of historical preparation text. Scientific methods, numerical rules and limitations are retained. Historical status statements below describe preparation, not the completed result. Original and derivative hashes are mapped in the package provenance ledger.

# Pre-data review corrections, protocol v2

The independently reviewed v1 manifest had SHA-256
93761f769ad64d1df31d3b899f8d283c5b6bc1555bceeea1e2ad565e967566e3.
No real-data stage occurred under v1 or v2.

Before any data access, the reviewer demonstrated that finite but extremely
large prediction inputs could overflow to infinity and silently be classified
by argmax. The numerical kernels now reject overflow/invalid arithmetic and
nonfinite outputs. Regressions cover fitting, geometry, projection and scoring.

The normal-equation integrity check now has an explicit returned diagnostic,
elementwise residual normalization and fixed 1e-10 tolerance, separate from
the 1e-9 objective-gradient tolerance.

Interval and decision labels now explicitly describe a case-resampling
sensitivity rule. They do not claim nominal confidence coverage, calibrated
noninferiority or statistical exclusion. This correction reflects historical
rank selection/test exposure and unknown writer/case dependence, rather than
changing a result: the real-data result remains unmeasured.

Negative-coordinate diagnostics now explicitly restore the training mean
before checking both <0 and <-1e-10. Their coordinate/case denominators are
frozen separately. Two regressions prevent confusing centering with a ReLU
support violation and distinguish numerical roundoff.

All 26 synthetic tests pass. REVIEW-RED-tests.txt and SUPPORT-RED-tests.txt preserve the regression
failures observed before these corrections; GREEN-tests.txt is the full
corrected suite. The fixed scientific margin, seeds, rank, heads, penalty,
denominator and resource limits are unchanged.
