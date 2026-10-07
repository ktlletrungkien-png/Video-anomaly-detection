# RWF-2000 Phase 2 — ResNet18 Temporal-Average Baseline

## Trạng thái

Phase 2 hoàn thành trên Kaggle ngày 07/10/2026. Gate A-G đều pass. Đây là kết quả baseline đã đo, không phải target hoặc ước lượng.

Không chạy lại training hoặc held-out test chỉ để cải thiện con số. Threshold và model đã được khóa trước final test.

## Provenance

| Trường | Giá trị |
|---|---|
| Git commit | `2af6ebdbb7b2617ce980b67fec20555b5d55220e` |
| Manifest SHA-256 | `a4215a6c67418b8c3c3a7c806bec64cfbeaeb39765c38b5000880898453df3b2` |
| Checkpoint SHA-256 | `c1e16c3aa93d560978c225bc213f2a42ab51a16255caed801602d40f0c7c6078` |
| Threshold artifact SHA-256 | `81fa9d82ffd4abbaa1d6bf5301f7a9b56bdc0e1091e35c31aaa0a9e3af17cda2` |
| Seed | `42` |
| Device | Tesla T4 / CUDA |
| PyTorch | `2.11.0+cu128` |
| Torchvision | `0.26.0+cu128` |

Checkpoint `best_checkpoint.pth` không nằm trong Git. Nó được giữ trong backup artifact riêng và phải được nhận dạng bằng SHA-256 ở trên.

## Dataset protocol

| Source | Derived split | Số clip | Vai trò |
|---|---|---:|---|
| `train` | `train` | 1.280 | Fit model |
| `train` | `val` | 320 | Chọn checkpoint và threshold |
| `val` | `test` | 400 | Held-out final evaluation |

Label: Non-violence `0`, Violence `1`. Không tune bằng derived test.

## Model và training config

- Pretrained ResNet18, frozen backbone.
- 16 RGB frames/clip, direct resize `224 x 224`.
- ImageNet normalization và clip-consistent training augmentation.
- Feature mỗi frame: 512.
- Temporal mean pooling.
- Classifier `Linear(512 -> 1)` trả raw logit.
- `BCEWithLogitsLoss`, AdamW.
- 5 epochs, batch size 4, learning rate `0.001`, weight decay `0.0001`.
- Best checkpoint: epoch 5 theo minimum validation loss.

Canonical config: `configs/rwf2000_resnet18_avg_kaggle_full.json`.

## Training history

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.609628 | 0.547322 |
| 2 | 0.541677 | 0.540639 |
| 3 | 0.515670 | 0.497823 |
| 4 | 0.492491 | 0.487546 |
| 5 | 0.473387 | 0.485344 |

## Validation result

Threshold được chọn trên derived validation: `0.4555857181549072`.

Rule: maximize Violence F1; tie-break bằng Violence recall; nếu vẫn hòa thì chọn threshold nhỏ nhất.

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.743750 |
| Precision | 0.693069 |
| Recall | 0.875000 |
| F1 | 0.773481 |
| ROC-AUC | 0.846133 |
| Evaluation loss | 0.485344 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[98, 62],
 [20, 140]]
```

## Final held-out test result

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.740000 |
| Precision | 0.712389 |
| Recall | 0.805000 |
| F1 | 0.755869 |
| ROC-AUC | 0.814525 |
| Evaluation loss | 0.526763 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[135, 65],
 [39, 161]]
```

Có 65 false positives và 39 false negatives. Threshold ưu tiên recall được chọn từ validation, vì vậy số false positives cao hơn false negatives là một trade-off cần được phân tích, không được sửa bằng test labels.

Accuracy 74% thấp hơn planning target tối thiểu 75% đúng 1 điểm phần trăm. Baseline vẫn hoàn thành vai trò đối chứng cho Temporal Transformer; không được mô tả là đã đạt target 75-85%.

## Kiểm tra local sau khi tải artifact

Ngày 07/10/2026, các kiểm tra sau đã pass:

- Toàn bộ local suite: `69 passed`; `pip check` sạch.
- Parse toàn bộ JSON/CSV artifacts.
- Tính lại Accuracy, Precision, Recall, F1 và ROC-AUC từ prediction CSV; khớp JSON.
- Tính lại BCE loss từ logits; chỉ có sai số float32 dưới `1e-8`.
- Xác nhận threshold và `predicted_label` nhất quán cho mọi row.
- Xác nhận 320 validation rows và 400 test rows, mỗi split cân bằng hai lớp.
- Không có `clip_id` trùng giữa validation và test.
- Threshold SHA-256 trong `test_metrics.json` khớp file local.
- Confusion-matrix image mở được và có đúng `[[135,65],[39,161]]`.

## Artifact inventory

- `environment.json`
- `history.csv`
- `run_config.json`
- `selected_threshold.json`
- `validation_metrics.json`
- `validation_predictions.csv`
- `test_metrics.json`
- `test_predictions.csv`
- `confusion_matrix.png`

Notebook `rwf2000-phase-2-baseline.ipynb` hiện ở workspace không được đưa vào commit kết quả này vì bản local được export trước full training/final evaluation: cell full-run chưa có execution output. Cần export lại đúng Kaggle saved version nếu muốn lưu notebook làm evidence.

## Giới hạn còn lại

- Chưa xác minh duplicate theo source video/provenance ngoài relative-path và clip ID.
- Direct resize có thể làm méo aspect ratio.
- Baseline temporal mean không mô hình hóa thứ tự/chuyển động; đây là lý do thực hiện Phase 3 Temporal Transformer.
- Chưa có qualitative review nội dung video cho các false positive/false negative hàng đầu.
