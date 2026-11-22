# Aerotrust v5 — Dataset Audit Conclusion

Audit timestamp: 2026-11-21T05:38:50.345291+00:00

Dataset: `/local_disk0/tmp/aerotrust/aerotrust_v5_resplit_local`

## 1. Dataset overview

- Total images: 9,350
- Total valid annotations: 20,471
- Test lock: **MATCH**
- Exact duplicate records: 0
- Cross-split exact duplicate records: 0
- Near-duplicate candidates: 4,386
- Cross-split near-duplicate candidates: 306
- Annotation errors: 0
- Metadata files identified: 0

## 2. Split distribution

| split   |   images |   image_share |   objects |   objects_per_image |   empty_or_missing_label_images |   multi_class_images |   invalid_label_lines |   invalid_class_ids |   invalid_coordinates |   zero_area_boxes |   mean_width |   mean_height |
|:--------|---------:|--------------:|----------:|--------------------:|--------------------------------:|---------------------:|----------------------:|--------------------:|----------------------:|------------------:|-------------:|--------------:|
| train   |     6515 |      0.696791 |     14305 |             2.1957  |                               0 |                  911 |                     0 |                   0 |                     0 |                 0 |      852.697 |       776.767 |
| val     |     1430 |      0.152941 |      3071 |             2.14755 |                               0 |                  176 |                     0 |                   0 |                     0 |                 0 |      887.129 |       800.524 |
| test    |     1405 |      0.150267 |      3095 |             2.20285 |                               0 |                  187 |                     0 |                   0 |                     0 |                 0 |      891.033 |       795.465 |

## 3. Class distribution

| class_name      |   test |   train |   val |
|:----------------|-------:|--------:|------:|
| Crack           |    281 |    1273 |   273 |
| Dent            |    835 |    3888 |   830 |
| Fastener Damage |    511 |    2395 |   519 |
| Missing rivet   |    717 |    3234 |   697 |
| Paint-off       |    239 |    1129 |   242 |
| Rupture         |    512 |    2386 |   510 |

## 4. Defect-size distribution

| size_bin   |   test |   train |   val |
|:-----------|-------:|--------:|------:|
| 0.1-0.25%  |    257 |    1180 |   264 |
| 0.25-1%    |    545 |    2545 |   545 |
| <0.1%      |    715 |    3214 |   692 |
| >=1%       |   1578 |    7366 |  1570 |

## 5. Distribution divergence

| comparison                       |         PSI |
|:---------------------------------|------------:|
| train_vs_val_class_distribution  | 3.02487e-05 |
| train_vs_val_size_distribution   | 0.000173115 |
| train_vs_test_class_distribution | 0.000268131 |
| train_vs_test_size_distribution  | 0.000248995 |

## 6. Positive findings

- Locked test fingerprint matches the current test images.
- No exact SHA-256 image duplicates were detected across splits.
- No malformed, invalid-class, invalid-coordinate or zero-area annotations were detected.
- 22.6% of valid annotations fall below 0.1% image area.
- Training class largest/smallest ratio is 3.4x.

## 7. Risks and issues

- 306 perceptual near-duplicate candidates cross split boundaries; manually inspect them.
- No obvious aircraft/inspection metadata files were discovered by filename/content keyword scan.

## 8. Recommendations

- Folder names repeat between splits. Determine whether these are meaningful aircraft/session identifiers.
- Do not modify the locked v5 test split before completing the audit and preserving the current baseline.
- If aircraft/session IDs exist, create a separate group-held-out evaluation split rather than randomly repartitioning images.
- If group IDs do not exist, perform exact and near-duplicate review and document that aircraft-independent generalization cannot be verified.
- For the next training ablation, preserve the current split and compare standard sampling against scale-aware sampling.
- Use validation metrics to select models and inference settings; do not repeatedly tune against the test set.

## 9. Interpretation limitations

- A matching image hash does not establish aircraft/session independence.
- Perceptual-hash candidates require manual visual review.
- PSI is a descriptive screening statistic, not a hypothesis test.
- Folder names and filename prefixes are candidate grouping clues, not verified aircraft IDs.
- Relative bbox area is not physical defect size in millimetres.
- This audit does not establish annotation correctness against the original aircraft imagery.
