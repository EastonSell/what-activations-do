# What this package can reproduce

## 1. Public checks: no real data required

The package is relocatable. It need not sit beside any private study directory.
Use Python 3.12 (verified with 3.12.14), NumPy 2.3.5 and threadpoolctl 3.6.0.
`requirements.txt` records these Python dependencies; packaging did not install
software. A compatible NumPy/BLAS runtime must already be available. The full
operational suite requires Linux `/proc` and POSIX process-group behavior.

From this package's root, set conservative thread limits and suppress bytecode
files so the frozen tree stays unchanged:

```sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
```

Check the whole public payload, including exact file coverage, and then verify
the separately frozen public source and baked protocol binding:

```sh
python -B -c "import sys; from pathlib import Path; sys.path.insert(0, 'source'); import custody, settings; custody.verify_manifest(Path('.'), 'PACKAGE_MANIFEST.json'); print('Public source:', settings.verify_source()); print('Public protocol:', settings.PROTOCOL_HASH)"
```

Run every public suite:

```sh
python -B -m unittest discover -s tests -v
(cd source && python -B -m unittest discover -s tests -v)
(cd source/protocol_v2 && python -B -m unittest discover -s tests -v)
python -B review/test_independent_audit.py
python -B review/independent_custody_checks.py
python -B review/independent_audit_operational_checks.py
```

The first suite verifies exact published report hashes, manifest provenance,
identity separation, unchanged scientific code/configuration, complete CSV
coverage, integer denominators and saved diagnostic bounds. It does not rerun
the real bootstrap or reconstruct real predictions. Hash consistency alone is
not an external attestation of scientific truth.

The remaining suites manufacture one-hot or other invented arrays and temporary
archives. They exercise numerical kernels, stage separation, subprocess resource
accounting, corruption rejection and read-only audit behavior. They do not fetch
the digit dataset, use original checkpoints or read owner-retained evidence.
Synthetic fitting/resampling is test arithmetic, not a new scientific experiment.
Synthetic pass counts are not additional scientific replications or arms.

Run the integrity command again after tests. The current checkout's logs should
be written outside the frozen package. `validation/` contains the preparation
run's saved evidence, including a second-checkout run. Test timings vary.

## 2. Exact report and the public derivative are different identities

The four files in `report/` are byte-identical to the final reviewed deliverables:

| File | SHA-256 |
|---|---|
| RESULTS.md | 5ae4420110292b65962febecc422bb221511c5131c09b42258adfc610be46e5e |
| RESULTS.json | 29ecf13e124180fd32ce9ac73efe091e40ee51da0a4890b2aa6d9193185bac70 |
| ALL_CONDITION_METRICS.csv | 9a005a4043500d65826c511f1897f77d9030bfdcf0cc55fc946813b36f733569 |
| ALL_FIT_DIAGNOSTICS.csv | b1174cbb5205fcf820b9e145d283acf7ae0093b901002475bc7a42278988b007 |

The original executed source-manifest hash is
`b169b28b144a511770743782e85f05ac008e09553e3845dab654a88e2604f528`;
the original protocol-manifest hash is
`061e722fb16c8ac0962307094b713c974658f9a049ac070f9c8f9cdfe709595a`.
The report correctly continues to cite those original identities.

The public copy has different source/protocol manifest identities because private
context was removed from ancillary documents and historical synthetic logs. The
new protocol identity is baked into `source/settings.py`; no identity or custody
check was weakened. `provenance/DERIVATIVE_LEDGER.json` maps both freezes and
every copied file. The two exact original manifest files preserve provenance,
but do not make the omitted original payload publicly available.

All original numerical code, configs, pipeline/auditor/watchdog code and tests
are byte-identical, except for the one protocol-hash literal in settings.py.
Three independent review probes change only their relative source-directory
locator. New public summary/provenance tests are publication-preparation tools.
Historical protocol/config status text remains historical, not a new claim that
the completed study is still awaiting execution. The exact report qualifies the
protocol's ambient-rank shorthand with actual measured design ranks.

Old clearance files cannot validate the derivative. Prior review decisions apply
only to their recorded original bytes, not automatically to sanitized documents,
new manifests or the new settings identity. No production clearance is supplied.
Do not bypass guards, relabel old artifacts, reseal evidence or substitute hashes
to make the derivative impersonate the original execution.

## 3. Numerical reproduction requires exact retained inputs

Real-data reproduction is not possible from this package alone. The original
study's five trained models, exact arrays and case order are not included. The
required historical inputs are owner-retained:

| Input identity | SHA-256 |
|---|---|
| Historical training archive | 79f4f5e467e65bc8e0571e9f68414b03f9deb54e2067fc2988537422050cd548 |
| Historical training manifest | eee5fe23598de00c5960e2cae9557b7655401eeb0411a6553e3d80e96b771399 |
| Historical held-out archive | 74843384a96f893b7c91d86f438a38cd02e37f26c18c5813c3a762a2692a2358 |
| Historical held-out manifest | 0f4096e4ce89c6190f21a2c67ddb98f0233e0f735a7d9adbd820eec211fcc8f8 |
| Exact dataset identity | f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443 |

Hashes are identity requirements, not download links, access rights or a promise
of public availability. Fetching another copy of the dataset, redrawing the split
or training replacement representations is not reproduction of this result.
Retraining with the existing public frozen-head source would be a fresh
replication with new artifacts/manifests, not an automatic way to satisfy the
original hardcoded archive hashes. Timings or archive metadata alone can change
bytes. Do not bypass hash gates to turn a newly reconstructed input into purported
original evidence.

A separately authorized reproduction must review and bind its exact source and
protocol identities, authenticate the exact retained inputs, use fresh output
directories, preserve all failures, fit the fixed 90 readouts once, pass the
training gates, obtain separate held-out admission and independently audit the
complete outputs under the fixed budgets. Archived historical clearance records
are evidence of that run, not standing permission for a new run. Neither this
document nor a passing test supplies such permission.

The public derivative can be considered for a separately reviewed reproduction
with new exact identity bindings. It is not the source executed for the reported
result. To reproduce using the original executed source identity, obtain the
exact immutable original source payload and verify it against the original
manifest before starting. Preserve the prescribed floating operation order and
runtime: strict sensitivity-rule boundaries can depend on sub-ulp arithmetic.
Different timing/resource receipts make new artifact/report hashes different;
bitwise cross-platform equality is not guaranteed.

## 4. Auditing original sealed evidence requires original provenance

The original saved stages are identified in the exact report:

| Stage manifest | SHA-256 |
|---|---|
| Training | bf3d3ba9d24e4ab293d3a69e33421c79cf6c4f50dbc54e97f6b53e85b2481fe5 |
| Training audit | 00f2f6660dc7ae31e529ce2dd544275b00b828346d51f13ef0d529e82728c99b |
| Held-out | b87755cc1b1a1ad1b32a2741979554122488446052a00bfac375334539f80a1f |
| Held-out audit | 6f30e1924d377b9deb133e06f244d8d373682c9ea56265084533332fba73986a |

Those stages bind the original source, protocol, dependencies and input hashes.
A full independent original-evidence audit requires authorized access to the
complete sealed artifacts, original immutable source and relevant provenance,
not just the report or public derivative. The supplied derivative deliberately
does not silently validate original production artifacts under a new identity.

The owner separately retains full original evidence and its reviewed provenance;
backup round-trip verification was completed before this package was prepared.
This package publishes neither storage identifiers nor private authorization,
access-log or stage-metadata records. It provides no public retrieval guarantee
or indefinite-retention promise. The public receipt summarizes prior independent
computational/source/report checks, not external peer review or public access to
the original arrays.

The original report builder was not copied because it is bound to the private
original evidence layout. Its SHA-256 is
`bf4a736fbfe31adbe6b6d25df949a56bd84a8af7f88f66696fb36ad7a7578cb1`.
The original final report check regenerated all four deliverables byte-identically
from sealed summaries; its public summary receipt is in `review/`. Repeating that
original document rebuild needs the retained summaries, resource/file-size
evidence, exact original manifests and original builder. Public CSV arithmetic
checks alone do not replace it or the full-array audits.
