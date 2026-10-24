# Frozen-head digit ablation: completed controlled experiment

**Result:** In five trained digit classifiers, removing leading hidden principal components hurt the original frozen output head more than paired, equal-norm random-orientation perturbations. The predeclared primary excess was **1.326 nats** of cross-entropy, with a conditional 95% case-bootstrap interval of **[1.247, 1.404]**. All five seed effects were positive.

This is evidence of **frozen-head perturbation sensitivity in this small benchmark**, not a semantic mechanism, natural-manifold intervention, irreversible information loss, biology, or novelty claim. The trained-versus-random-hidden-feature comparison **failed its competence gate and is ineligible**.

Start with [the result and limitations](report/RESULTS.md), [dense curves](report/frozen-head-curves.png), and the unchanged [pre-execution protocol](source/PROTOCOL.md).

## Contents

- `source/`: exact 14-file pre-execution source freeze and its manifest. Its original README/protocol status language intentionally records preparation time, not the completed status of this package.
- `report/`: completed result, all-rank figure, and machine-readable presentation summary.
- `receipts/`: all 10,890 descriptive curve rows, training/held-out summaries and manifests, resource logs, independent numerical checks, and review receipts.
- `tools/`: read-only audit adapter, portable direct checker, and chart generator. These are separate from the frozen producer source.
- `EVIDENCE.json`: raw archive hashes, sizes, and retention status.
- `PUBLICATION_MANIFEST.json`: package file hashes, added when the reviewed package is sealed.

The full source, training, and approximately 315 MB held-out archives were retained separately and verified by full-byte readback from durable storage. **They are not public download links and are not bundled here.** This compact package permits source inspection and inspection/replotting of reported curves, but complete independent rescoring requires the retained raw arrays. Hashes alone do not provide access or prove experimental provenance.

## Read-only verification

The recorded source freeze is `10a4ea2c6083e5f248cd4cd211b83ec6530f71e9e11591df9ee2f1d4e7ebf117`.
Synthetic tests require NumPy and scikit-learn (the execution used 2.3.5 and 1.8.0):

```sh
cd source
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

With the retained training and held-out directories available, the portable checker can reproduce the independent direct checks without training or running the producer:

```sh
python tools/direct_check.py --source source --training TRAINING_DIR --heldout HELDOUT_DIR --out NEW_CHECK.json
```

The frozen full audit also reconstructs every intervention and verifies head/geometry binding. `tools/audit_cached.py --help` describes its invocation; it only avoids repeated decompression and does not change the frozen audit's numerical logic. The archived receipts document the completed audit rather than requesting another experimental run.

This separate study follows the public repository snapshot [baa500a](https://github.com/EastonSell/what-activations-do/tree/baa500a5567b3845513edb52d9b241edbeb45bab). No other experiment was resumed. Code is MIT-licensed; dataset attribution is in the result report.
