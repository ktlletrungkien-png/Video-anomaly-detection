# RWF-2000 Phase 3 — ResNet18 Temporal Transformer

## Trạng thái

Phase 3 full run và fixed-threshold held-out evaluation hoàn thành trên Kaggle ngày 08/10/2026. Đây là kết quả đã đo của ablation đầu tiên, không phải target hoặc ước lượng.

Model, checkpoint và threshold đã được khóa trước final test. Không retune bằng derived-test labels và không chạy lại final test chỉ để chọn kết quả đẹp hơn.

## Provenance

| Trường | Giá trị |
|---|---|
| Git commit chạy trên Kaggle | `ff59202d4e8db05509c0d6fe916b0e9661beba7e` |
| Manifest SHA-256 | `a4215a6c67418b8c3c3a7c806bec64cfbeaeb39765c38b5000880898453df3b2` |
| Checkpoint SHA-256 | `4fcb2038d5aa485e500cd17dcfc254f258b7469801e26ad8549ae6bb1e2c7001` |
| Threshold artifact SHA-256 | `3ca16b784b83b3b8e96363081d789310901716395fa38173d9759bb5f0baab10` |
| Validation predictions SHA-256 | `34bb185461c2ab257e0ad612fc8e808050757b80e801029dc1cdc3dd84e13488` |
| Seed | `42` |
| Device | Tesla T4 / CUDA 12.8 |
| PyTorch | `2.11.0+cu128` |
| Torchvision | `0.26.0+cu128` |

Checkpoint `best_checkpoint.pth` không nằm trong Git và hiện không có trong thư mục artifact local này. Bản backup bên ngoài Git phải được nhận dạng bằng SHA-256 ở trên.

## Dataset protocol

| Source | Derived split | Số clip | Vai trò |
|---|---|---:|---|
| `train` | `train` | 1.280 | Fit model |
| `train` | `val` | 320 | Chọn checkpoint và threshold |
| `val` | `test` | 400 | Held-out final evaluation |

Label: Non-violence `0`, Violence `1`. Không tune bằng derived test.

## Model và training config

- Pretrained ResNet18, frozen backbone và frozen BatchNorm.
- 16 RGB frames/clip, direct resize `224 x 224`.
- ImageNet normalization và clip-consistent training augmentation.
- Feature mỗi frame: 512.
- Sinusoidal positional encoding.
- Transformer Encoder: 2 layers, 4 heads, FFN 1024, dropout 0.1, pre-norm, GELU.
- Temporal mean pooling sau Transformer.
- Classifier `Linear(512 -> 1)` trả raw logit.
- `BCEWithLogitsLoss`, AdamW.
- 5 epochs, batch size 4, learning rate `0.001`, weight decay `0.0001`.
- Best checkpoint: epoch 4 theo minimum validation loss.

Canonical config: `configs/rwf2000_resnet18_transformer_kaggle_full.json`.

## Training history

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 0.762497 | 0.496018 |
| 2 | 0.551234 | 0.563476 |
| 3 | 0.557637 | 0.486039 |
| 4 | 0.507188 | **0.468445** |
| 5 | 0.480668 | 0.530346 |

Epoch 4 là checkpoint tốt nhất. Epoch 5 tiếp tục giảm train loss nhưng validation loss tăng, nên không được chọn.

## Validation result

Threshold được chọn trên derived validation: `0.17900283634662628`.

Rule: maximize Violence F1; tie-break bằng Violence recall; nếu vẫn hòa thì chọn threshold nhỏ nhất.

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.756250 |
| Precision | 0.683036 |
| Recall | 0.956250 |
| F1 | 0.796875 |
| ROC-AUC | 0.866367 |
| Evaluation loss | 0.468445 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[89, 71],
 [ 7,153]]
```

## Final held-out test result

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.747500 |
| Precision | 0.684015 |
| Recall | 0.920000 |
| F1 | 0.784648 |
| ROC-AUC | 0.838975 |
| Evaluation loss | 0.538910 |

Confusion matrix `[[TN, FP], [FN, TP]]`:

```text
[[115, 85],
 [ 16,184]]
```

Có 85 false positives và 16 false negatives. Threshold validation ưu tiên F1/recall nên model bắt được 92% clip Violence nhưng tạo nhiều cảnh báo giả.

## So sánh với Phase 2 baseline

| Metric | Baseline | Transformer | Delta Transformer − baseline |
|---|---:|---:|---:|
| Accuracy | 0.740000 | 0.747500 | +0.007500 |
| Precision | 0.712389 | 0.684015 | -0.028375 |
| Recall | 0.805000 | 0.920000 | +0.115000 |
| F1 | 0.755869 | 0.784648 | +0.028780 |
| ROC-AUC | 0.814525 | 0.838975 | +0.024450 |
| False positives | 65 | 85 | +20 |
| False negatives | 39 | 16 | -23 |

Transformer cải thiện Recall, F1 và ROC-AUC nhưng chỉ tăng Accuracy 0,75 điểm phần trăm và làm Precision giảm. Accuracy 74,75% vẫn thấp hơn planning target tối thiểu 75% đúng 0,25 điểm phần trăm và thấp hơn mục tiêu 82–88%.

Kết quả này cho thấy temporal attention có ích cho việc giảm bỏ sót Violence, nhưng frozen ImageNet backbone, RGB-only frame features và mean pooling vẫn giới hạn khả năng phân biệt chuyển động bạo lực với chuyển động mạnh không bạo lực.

## Kiểm tra local sau khi tải artifact

Ngày 08/10/2026, các kiểm tra sau đã pass:

- Parse toàn bộ JSON/CSV artifacts.
- Tính lại Accuracy, Precision, Recall, F1 và confusion matrix từ 400 test predictions; khớp `test_metrics.json`.
- Xác nhận threshold trong prediction CSV khớp `selected_threshold.json`.
- Xác nhận SHA-256 của `selected_threshold.json` khớp `test_metrics.json`.
- Xác nhận SHA-256 của `validation_predictions.csv` khớp threshold artifact.
- Xác nhận manifest, architecture contract, checkpoint ID và threshold provenance nhất quán.
- Xác nhận 320 validation clips và 400 test clips, mỗi split cân bằng hai lớp.

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

Không commit dataset hoặc `best_checkpoint.pth`.

## Kết luận và giới hạn

- Phase 3 là ablation hợp lệ và cải thiện các metric ưu tiên an toàn, nhưng chưa đạt target Accuracy.
- Không được đổi threshold bằng test labels để tăng Accuracy.
- Nếu thực hiện partial backbone fine-tuning sau khi đã xem test, phải ghi rõ đó là post-hoc exploratory follow-up; kết quả Phase 3 hiện tại vẫn là held-out result chính.
- Chưa có qualitative review nội dung của các false positive/false negative hàng đầu.
- Chưa đo inference latency/FPS cho artifact này.
