# Phase 3 — RWF-2000 Temporal Transformer

## Trạng thái — 08/10/2026

Phần implementation và validation không cần dataset thật đã hoàn thành trên local:

- Đã thêm `ResNet18TemporalTransformer` với sinusoidal positional encoding, Transformer Encoder 2 lớp/4 heads, FFN 1024, dropout 0.1, pre-norm, temporal mean pooling và classifier `Linear(512 -> 1)`.
- Đã giữ backbone ResNet18 frozen và BatchNorm ở eval mode trong run đầu.
- Config, CLI train/evaluate và checkpoint provenance đã dispatch theo architecture; Transformer không dùng nhầm checkpoint baseline.
- Phase 3 config khóa manifest SHA-256 `a4215a6c67418b8c3c3a7c806bec64cfbeaeb39765c38b5000880898453df3b2`.
- Full local suite: `81 passed`, `7 warnings`. Các warning đều là thông báo PyTorch về nested-tensor optimization khi `norm_first=True`, không phải test failure.
- CPU synthetic smoke bằng ResNet18 `weights=None` đã pass: output `[2]` hữu hạn; backbone không có gradient; Transformer và classifier đều cập nhật sau một optimization step.
- Không chạy real-batch, overfit, pilot, full training hoặc held-out test trên local.

Full training và fixed-threshold held-out evaluation đã hoàn thành trên Kaggle bằng commit `ff59202d4e8db05509c0d6fe916b0e9661beba7e`.

Kết quả derived test chính thức:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.747500 |
| Precision | 0.684015 |
| Recall | 0.920000 |
| F1 | 0.784648 |
| ROC-AUC | 0.838975 |

Threshold Transformer được chọn từ validation là `0.17900283634662628`; confusion matrix test là `[[115,85],[16,184]]`. Best checkpoint ở epoch 4, SHA-256 `4fcb2038d5aa485e500cd17dcfc254f258b7469801e26ad8549ae6bb1e2c7001`.

So với baseline, Transformer tăng Accuracy `0.0075`, Recall `0.115`, F1 `0.028780` và ROC-AUC `0.024450`, nhưng Precision giảm `0.028375`. Kết quả chưa đạt planning target Accuracy; không được retune bằng test labels.

Source of truth cho artifacts và phân tích: `results/rwf-2000/phase3_resnet18_temporal_transformer/README.md`.

## 0. Điểm xuất phát đã khóa

Phase 2 ResNet18 + temporal average baseline đã hoàn thành trên Kaggle ngày 07/10/2026.

Baseline held-out result:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.740000 |
| Precision | 0.712389 |
| Recall | 0.805000 |
| F1 | 0.755869 |
| ROC-AUC | 0.814525 |

Threshold baseline `0.4555857181549072` chỉ thuộc baseline. Phase 3 phải chọn threshold mới bằng derived validation của chính Transformer. Không được dùng baseline test errors hoặc Transformer test labels để chọn hyperparameter.

Source of truth cho baseline artifacts: `results/rwf-2000/phase2_resnet18_temporal_avg/README.md`.

## 1. Mục tiêu

Xây mô hình chính cho nhánh violence:

```text
[B,T,3,H,W]
-> pretrained ResNet18 frame features [B,T,512]
-> positional encoding
-> 2-layer Temporal Transformer, 4 heads
-> temporal mean pooling
-> Linear(512 -> 1)
-> raw violence logits [B]
```

Phase 3 là controlled ablation so với baseline. Dataset, manifest, preprocessing, frame count, resolution, label mapping, loss, checkpoint rule và threshold protocol phải giữ giống Phase 2 trừ thành phần temporal encoder được khai báo rõ.

Mô hình không bắt buộc phải tốt hơn baseline để thí nghiệm có giá trị. Nếu không cải thiện, phải báo cáo trung thực và phân tích nguyên nhân.

## 2. Protocol không được thay đổi

- Manifest SHA-256: `a4215a6c67418b8c3c3a7c806bec64cfbeaeb39765c38b5000880898453df3b2`.
- Seed: `42`.
- Derived train: 1.280 clip từ source `train`.
- Derived validation: 320 clip từ source `train`.
- Derived test: 400 clip từ source `val`, giữ khóa đến final evaluation.
- Non-violence `0`, Violence `1`.
- 16 uniformly sampled ordered frames.
- Direct resize `224 x 224`, ImageNet normalization.
- BCEWithLogitsLoss và AdamW.
- Best checkpoint theo minimum validation loss, earliest epoch khi hòa.
- Threshold theo validation F1, rồi recall, rồi threshold nhỏ nhất.
- Không tune bằng test labels và không chạy lại test chỉ để chọn kết quả đẹp hơn.

## 3. Kiến trúc khởi đầu

| Thành phần | Giá trị khởi đầu |
|---|---|
| Backbone | ResNet18 ImageNet pretrained |
| Backbone state | Frozen trong run đầu |
| Feature dimension | 512 |
| Sequence length | 16 |
| Positional encoding | Learned hoặc sinusoidal, chọn một và khóa trước pilot |
| Transformer layers | 2 |
| Attention heads | 4 |
| Feed-forward dimension | 1024 |
| Dropout | 0.1 |
| Layout | `batch_first=True`, pre-norm ưu tiên |
| Temporal pooling | Mean sau Transformer |
| Classifier | Linear 512 -> 1 |

Mean pooling sau Transformer giữ classifier/pooling gần baseline nhất, giúp so sánh tập trung vào temporal self-attention. Không thêm CLS token, optical flow, YOLO hoặc augmentation mới trong ablation đầu tiên.

## 4. Implementation local

### Bước 1 — Model module

Thêm module riêng, ví dụ:

```text
src/rwf2000/models/resnet18_temporal_transformer.py
tests/test_temporal_transformer_model.py
```

Acceptance criteria:

- Input `[B,T,3,H,W]`, output raw logits `[B]`, kể cả `B=1`.
- Positional information thực sự được thêm trước Transformer.
- Transformer nhận đúng `[B,T,512]`; không trộn batch/time.
- Frozen backbone và BatchNorm giữ eval mode.
- Classifier và Transformer trainable.
- Tests dùng `weights=None`, không cần Internet.
- Có test chứng minh đổi thứ tự frame có thể thay đổi output khi positional encoding bật.

### Bước 2 — Config và checkpoint contract

Mở rộng config có version/architecture rõ ràng:

- `architecture=ResNet18TemporalTransformer`.
- `feature_dim=512`.
- `num_layers=2`, `nhead=4`, `dim_feedforward=1024`, `dropout=0.1`.
- Positional-encoding type và max sequence length.
- Không làm checkpoint Transformer tương thích giả với baseline checkpoint.

### Bước 3 — Reuse training/evaluation foundation

Tái sử dụng:

- Dataset/DataLoader và epoch-aware augmentation.
- BCEWithLogitsLoss, AdamW, seeding.
- Validation-only checkpoint/threshold selection.
- Provenance guards và fixed-threshold final evaluation.

Chỉ generalize các architecture checks cần thiết; không làm yếu manifest/checkpoint/threshold linkage đã có.

### Bước 4 — Local validation

- Unit tests toàn repository pass.
- CPU synthetic forward pass.
- One optimization step cập nhật Transformer/classifier nhưng không cập nhật frozen backbone.
- Checkpoint round-trip trả logits giống nhau.
- Train/eval preprocessing contract giống baseline.
- Không chạy full dataset local.

## 5. Kaggle gates

### Gate A — Tests và environment

- Checkout đúng commit Phase 3.
- Ghi lại PyTorch/Torchvision/CUDA/GPU.
- Toàn bộ tests pass.

### Gate B — Real-batch forward

- Derived train only.
- Kiểm tra logits shape/finite, GPU memory và throughput.

### Gate C — One-batch overfit

- Một batch cố định có cả hai lớp từ derived train.
- Loss phải giảm rõ.
- Không dùng validation/test.

### Gate D — Pilot 1 epoch

- Đo memory, runtime và DataLoader stability.
- Xác nhận checkpoint, validation metrics và threshold artifacts.
- Pilot metric không phải final result.

### Gate E — Full Transformer training

- Khóa config bằng validation evidence.
- Không mở derived test.
- Lưu best checkpoint và validation-selected threshold.

### Gate F — Final held-out evaluation

- Chạy đúng một lần sau khi model/config/threshold đã khóa.
- Lưu predictions, metrics, confusion matrix và provenance.
- Không retune sau khi xem test.

## 6. So sánh bắt buộc với baseline

| Thành phần | Baseline | Transformer |
|---|---|---|
| Manifest/split | Giống nhau | Giống baseline |
| Frames/resolution | 16 / 224² | Giống baseline |
| Backbone | Frozen ResNet18 | Frozen ResNet18 ở run đầu |
| Temporal module | Mean | Transformer 2L/4H + mean |
| Loss/optimizer | BCE / AdamW | Giống baseline |
| Checkpoint selection | Min val loss | Giống baseline |
| Threshold selection | Validation only | Validation only |

Báo cáo tối thiểu:

- Accuracy, Precision, Recall, F1, ROC-AUC.
- Confusion matrix.
- Validation loss history.
- Parameter count và inference latency nếu có thể.
- Chênh lệch tuyệt đối so với baseline cho từng metric.
- False positives/false negatives và ít nhất một qualitative example mỗi loại khi có video.

## 7. Quy tắc quyết định

- Chỉ unfreeze backbone như một controlled follow-up nếu frozen Transformer underfit theo train/validation evidence.
- Không đổi frame count, resolution, aspect-ratio policy hoặc augmentation cùng lúc với kiến trúc temporal trong ablation đầu.
- Không chọn checkpoint theo test accuracy.
- Nếu Transformer không cải thiện, giữ baseline làm phương án triển khai và báo cáo ablation trung thực.
- Sau khi Phase 3 ổn định, ưu tiên nhánh normal-profile Ped2/Avenue bắt buộc trước các mô hình anomaly lớn tùy chọn.

## 8. Definition of Done

- [x] Model contract và positional encoding có tests.
- [x] Full local suite pass.
- [ ] Real-batch forward pass.
- [ ] One-batch overfit pass.
- [ ] Pilot pass.
- [x] Full training hoàn thành trên derived train.
- [x] Best checkpoint và threshold chỉ dùng derived validation.
- [x] Final test chạy sau khi khóa config.
- [x] Artifacts nhẹ và provenance được lưu.
- [x] So sánh công bằng với baseline Phase 2.
- [x] Không có test-label leakage hoặc unsupported metric claims.

Ba gate real-batch, one-batch overfit và pilot chưa được đánh dấu vì log riêng của các gate này chưa được copy vào repo cùng full-run artifacts.
