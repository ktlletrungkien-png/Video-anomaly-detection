# Phase 2 — Kế hoạch ResNet18 + Temporal Average Pooling Baseline

## 0. Trạng thái hoàn tất — 07/10/2026

Phase 2 đã hoàn thành trên Kaggle. Gate A-G đều pass; không cần chạy lại training hoặc final test nếu không phát hiện lỗi thực sự.

Provenance chính:

- Git commit: `2af6ebdbb7b2617ce980b67fec20555b5d55220e`.
- Manifest SHA-256: `a4215a6c67418b8c3c3a7c806bec64cfbeaeb39765c38b5000880898453df3b2`.
- Checkpoint SHA-256: `c1e16c3aa93d560978c225bc213f2a42ab51a16255caed801602d40f0c7c6078`.
- Best epoch: 5; validation loss: `0.485344`.
- Validation-selected threshold: `0.4555857181549072`.
- Kaggle tests: 68 pass, 14 deprecation warnings.

Measured held-out test result:

| Accuracy | Precision | Recall | F1 | ROC-AUC | Loss |
|---:|---:|---:|---:|---:|---:|
| 0.740000 | 0.712389 | 0.805000 | 0.755869 | 0.814525 | 0.526763 |

Confusion matrix `[[TN,FP],[FN,TP]] = [[135,65],[39,161]]`.

Các CSV/JSON artifacts đã được kiểm tra lại ở local: metric tính lại khớp, threshold/predicted labels nhất quán, validation/test không trùng clip ID và threshold hash khớp provenance. Kết quả nhẹ nằm tại `results/rwf-2000/phase2_resnet18_temporal_avg/`; checkpoint không được commit.

Accuracy 74% thấp hơn planning target tối thiểu 75% một điểm phần trăm. Đây vẫn là baseline hợp lệ cho ablation, không phải lý do để retune bằng held-out test. Bước tiếp theo là Phase 3 Temporal Transformer theo `PHASE3_RWF2000_TEMPORAL_TRANSFORMER_PLAN.md`.

## 1. Mục tiêu

Phase 2 xây baseline bắt buộc cho nhánh phát hiện bạo lực RWF-2000:

```text
16 RGB frames
-> pretrained ResNet18 cho từng frame
-> frame features [B, T, 512]
-> temporal average pooling
-> clip feature [B, 512]
-> linear binary classifier
-> một violence logit cho mỗi clip
```

Phase này phải tạo được một baseline đúng, tái lập được và đủ làm đối chứng công bằng cho Temporal Transformer sau này.

Phase 2 không bao gồm:

- Temporal Transformer.
- Ped2/Avenue anomaly branch.
- UBI-Fights external test.
- Demo UI.
- YOLO/person detection.
- Tuning bằng held-out labels.
- Ablation aspect-ratio/letterbox trước khi baseline chuẩn hoạt động.

## 2. Protocol đã khóa từ Phase 1

Dataset root trên Kaggle:

```text
/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000
```

Manifest configuration:

| Trường | Giá trị |
|---|---|
| Schema | `rwf2000-manifest-v1` |
| Seed | `42` |
| Validation fraction | `0.2` |
| Label mapping | Non-violence `0`, Violence `1` |
| Frames/clip | `16` |
| Resolution | `224 x 224` direct resize |

Split protocol:

| Source | Derived use | Số clip | Cho phép dùng để quyết định model? |
|---|---|---:|---|
| source `train` | derived `train` | 1.280 | Có, để fit model |
| source `train` | derived `val` | 320 | Có, để model selection và threshold selection |
| source `val` | derived `test` | 400 | Không, giữ khóa đến final evaluation |

Nguyên tắc:

- Không regenerate manifest với seed/fraction khác chỉ để tìm split dễ hơn.
- Không dùng derived test để chọn epoch, learning rate, freeze policy hoặc threshold.
- Test chỉ được chạy sau khi baseline configuration và checkpoint-selection rule đã chốt bằng derived validation.
- Kết quả test phải được báo cáo tách khỏi validation.

## 3. Contract kiến trúc baseline

### 3.1. Input/output

| Tensor | Shape | Dtype |
|---|---|---|
| Input frames | `[B, T, 3, H, W]` | `float32` |
| Flattened frames | `[B*T, 3, H, W]` | `float32` |
| ResNet18 features | `[B*T, 512]` | `float32` |
| Temporal features | `[B, T, 512]` | `float32` |
| Mean-pooled feature | `[B, 512]` | `float32` |
| Output logits | `[B]` | `float32` |

Model `forward()` trả raw logits, không áp dụng sigmoid. Training dùng `BCEWithLogitsLoss`. Sigmoid chỉ dùng khi cần xác suất cho metric/inference.

### 3.2. Backbone

- Dùng `torchvision.models.resnet18` với pretrained ImageNet weights trong training thật.
- Bỏ classification head gốc và giữ feature dimension 512.
- Freeze toàn bộ backbone ở baseline đầu tiên; chỉ train linear classifier.
- Tests phải có khả năng tạo model với `weights=None` để không tải network asset.
- Unfreeze backbone chỉ là controlled follow-up nếu head-only baseline underfit và sanity checks đã pass.

### 3.3. Temporal aggregation

- Reshape feature về `[B, T, 512]`.
- Mean theo dimension `T`.
- Không thêm attention, LSTM, Transformer hoặc learned temporal weights.
- Không đảo hoặc shuffle frame order trong Dataset.

Mean pooling là permutation-invariant; đây là giới hạn có chủ đích của baseline và là lý do cần so sánh với Temporal Transformer sau này.

### 3.4. Preprocessing

- Giữ ImageNet mean/std hiện tại.
- Giữ 16 uniformly sampled ordered frames.
- Giữ direct resize `224 x 224` cho baseline đầu tiên.
- Giữ clip-consistent train augmentation: cùng transform parameters cho toàn bộ 16 frames.
- Không dùng augmentation theo thời gian.

Resolution thật đa dạng, vì vậy direct resize có thể làm méo aspect ratio. Đây là limitation cần ghi trong báo cáo; không âm thầm đổi sang letterbox trước khi có baseline chuẩn để so sánh.

## 4. Blocker nhỏ cần xử lý trước full training

### 4.1. Epoch-aware augmentation

Dataset hiện tạo RNG từ `seed + clip_id`, vì vậy cùng clip có thể nhận cùng augmentation ở mọi epoch. Điều này không làm sai evaluation nhưng hạn chế train augmentation.

Thay đổi bounded cần có trước full training:

- Dataset giữ `base_seed` và `epoch`.
- Có API `set_epoch(epoch)` hoặc cơ chế tương đương.
- Clip RNG phụ thuộc `base_seed`, `epoch` và `clip_id`.
- Cùng seed/epoch/clip phải tái lập.
- Khác epoch có thể sinh augmentation khác.
- Eval/inference tiếp tục deterministic và không augmentation.
- Nếu dùng worker processes, tránh `persistent_workers=True` cho đến khi epoch propagation được kiểm chứng.

### 4.2. Training dependencies

Khi implement baseline, chỉ thêm dependencies thực sự dùng:

- `torchvision` cho ResNet18 và pretrained weights.
- `scikit-learn` nếu dùng cho ROC-AUC, precision, recall, F1 và confusion matrix.

Không pin toàn bộ Kaggle environment. Version constraints phải tương thích với khai báo PyTorch hiện tại và được kiểm tra cả local lẫn Kaggle.

### 4.3. Notebook import order

Notebook phải cài project trước mọi `import rwf2000`. Nếu interactive kernel giữ module/path state cũ, restart session/kernel thay vì đưa Kaggle-specific path handling vào core package.

## 5. Kế hoạch implementation local

Mỗi bước dưới đây phải là một diff nhỏ, review được. Không bắt đầu bước sau khi bước trước chưa pass.

### Bước 1 — Thêm model module tối thiểu

File dự kiến:

```text
src/rwf2000/models/__init__.py
src/rwf2000/models/resnet18_temporal_average.py
tests/test_baseline_model.py
```

Nội dung:

- Model config tối thiểu.
- ResNet18 feature extractor.
- Freeze/unfreeze policy tường minh.
- Temporal mean pooling.
- Linear head 512 -> 1.
- Shape validation và lỗi rõ cho input không phải `[B,T,3,H,W]`.

Acceptance criteria:

- Forward CPU với synthetic tensor hoạt động.
- Output shape `[B]`.
- Output finite.
- Không trộn batch/time dimensions.
- Backbone frozen đúng khi cấu hình freeze.
- Classifier parameters trainable.
- Tests không cần Internet hoặc pretrained download.

### Bước 2 — Sửa augmentation seed theo epoch

File dự kiến:

```text
src/rwf2000/dataset.py
tests/test_dataset.py
```

Acceptance criteria:

- Cùng clip, seed và epoch cho cùng output augmentation.
- Đổi epoch thay đổi RNG stream.
- Evaluation output không đổi theo epoch.
- Temporal order và clip boundary tests vẫn pass.

### Bước 3 — Thêm training configuration

File dự kiến có thể gồm:

```text
src/rwf2000/train_config.py
configs/rwf2000_resnet18_avg_baseline.json
```

Config tối thiểu phải lưu:

- Manifest path và dataset root được truyền ngoài source.
- Seed.
- Frames/clip.
- Resolution.
- Batch size.
- Number of workers.
- Pretrained weights identifier.
- Freeze policy.
- Optimizer và learning rate.
- Weight decay.
- Epoch count.
- Checkpoint-selection metric.
- Git commit khi run.

Không hard-code Kaggle paths trong source/config được commit. Kaggle notebook truyền path runtime.

### Bước 4 — Thêm training loop tối thiểu

File dự kiến:

```text
src/rwf2000/train_baseline.py
tests/test_training_step.py
```

Training loop phải có:

- Explicit `train()` và `eval()` modes.
- `BCEWithLogitsLoss`.
- AdamW.
- Gradient zeroing/backprop/optimizer step đúng.
- Train loss và validation loss theo sample count.
- No-gradient validation.
- Deterministic seeds.
- Device selection từ CLI/config.
- Checkpoint chỉ dựa trên derived validation.
- Không load derived test trong training loop.

Checkpoint selector ban đầu:

- Chọn best checkpoint theo validation loss để không phụ thuộc threshold.
- Ghi validation metrics ở threshold 0.5 như diagnostics, không coi 0.5 là threshold cuối.

Acceptance criteria:

- Một synthetic optimization step làm parameter classifier thay đổi.
- Frozen backbone không nhận update.
- Train/eval mode được test.
- Checkpoint save/load khôi phục logits tương thích.

### Bước 5 — Thêm metric/evaluation foundation

File dự kiến:

```text
src/rwf2000/metrics.py
src/rwf2000/evaluate_baseline.py
tests/test_metrics.py
```

Metrics bắt buộc:

- Accuracy.
- Precision cho Violence.
- Recall cho Violence.
- F1 cho Violence.
- ROC-AUC.
- Confusion matrix.

Threshold protocol:

1. Thu validation probabilities từ best checkpoint.
2. Chọn threshold bằng derived validation theo rule đã khai báo, ví dụ maximize Violence F1.
3. Lưu threshold và validation evidence.
4. Áp dụng threshold cố định đúng một lần cho derived test.
5. Không điều chỉnh lại threshold sau khi xem test.

Edge cases cần test:

- Tất cả prediction cùng class.
- Batch size 1.
- Không chia cho zero khi không có predicted positive.
- ROC-AUC phải báo lỗi rõ nếu target chỉ có một class.

### Bước 6 — Thêm experiment output contract

Mỗi run phải ghi vào output directory được truyền từ CLI:

```text
run_config.json
environment.json
history.csv
best_checkpoint.pth
validation_predictions.csv
validation_metrics.json
selected_threshold.json
```

Final evaluation mới bổ sung:

```text
test_predictions.csv
test_metrics.json
confusion_matrix.png
```

Không commit checkpoint lớn. Chỉ commit code/config và có thể commit kết quả nhẹ sau khi review.

## 6. Validation theo từng tầng

### Gate A — Unit tests local

Chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Yêu cầu:

- Toàn bộ test Phase 1 tiếp tục pass.
- Model shape/freeze tests pass.
- Training-step tests pass.
- Metrics tests pass.

### Gate B — Model forward smoke test local

Input synthetic:

```text
[B=2, T=16, C=3, H=224, W=224]
```

Output:

```text
logits [2]
```

Mục đích: xác nhận wiring model mà không cần RWF-2000 local.

### Gate C — Real batch forward trên Kaggle

Sau khi code được commit/push và Kaggle pull:

- Bật GPU chỉ ở bước này hoặc khi one-batch overfit bắt đầu.
- Load một real batch từ derived train.
- Forward pass.
- Kiểm tra logits shape, finite values và GPU memory.
- Chưa chạy nhiều epoch.

### Gate D — One-batch overfit

Dùng một batch nhỏ chỉ từ derived train:

- Tắt hoặc cố định random augmentation cho sanity check.
- Lặp cùng batch trong số step giới hạn.
- Theo dõi loss giảm và prediction phù hợp label.
- Không dùng derived val/test.

Nếu model không overfit được một batch nhỏ, dừng và kiểm tra label, loss, logits, optimizer, freeze policy và preprocessing. Không chạy full training.

### Gate E — Short pilot run

Chạy 1-2 epochs trên derived train/val để đo:

- Throughput.
- GPU memory.
- DataLoader stability.
- Train/validation loss wiring.
- Checkpoint/resume behavior.

Pilot results là infrastructure evidence, không phải final model result.

### Gate F — Full baseline training

Chỉ chạy sau khi A-E pass:

- Train trên derived train.
- Model selection bằng derived validation.
- Lưu best checkpoint.
- Chọn threshold bằng derived validation.
- Không mở derived test trong quá trình tuning.

### Gate G — Final same-domain evaluation

Khi configuration đã khóa:

- Load best checkpoint.
- Load threshold đã chọn từ validation.
- Evaluate derived test đúng một lần cho baseline chính.
- Lưu predictions, metrics và confusion matrix.
- Báo cả failure cases, không chỉ metric tổng.

## 7. Cấu hình khởi đầu đề xuất

Đây là starting configuration, chưa phải kết quả hoặc hyperparameter tối ưu:

| Thành phần | Giá trị khởi đầu |
|---|---|
| Frames | 16 |
| Resolution | 224 x 224 |
| Backbone | ImageNet-pretrained ResNet18 |
| Backbone state | Frozen |
| Temporal pooling | Mean |
| Head | Linear 512 -> 1 |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Seed | 42 |
| Model selection | Minimum validation loss |
| Threshold selection | Derived validation only |

Learning rate, batch size, weight decay, epochs và number of workers phải được chọn sau one-batch/pilot evidence và được lưu trong config. Không ghi giá trị tùy tiện thành protocol đã khóa trước khi đo tài nguyên Kaggle.

## 8. Kế hoạch Kaggle cho Phase 2

Sau khi implementation local được test, commit và push:

1. Tạo Kaggle Notebook mới cho baseline hoặc fork notebook Phase 1.
2. Bật GPU.
3. Attach cùng RWF-2000 dataset và Phase 1 artifacts nếu cần.
4. Clone/pull đúng Git commit.
5. Cài project trước khi import package.
6. Chạy unit tests.
7. Xác nhận GPU và versions.
8. Tạo hoặc load manifest với seed/fraction đã khóa.
9. Chạy real batch forward.
10. Chạy one-batch overfit.
11. Chạy pilot 1-2 epochs.
12. Review memory, throughput và logs.
13. Chỉ sau đó chạy full baseline.
14. Save & Run All với output directory dưới `/kaggle/working`.

## 9. Definition of Done cho Phase 2

Phase 2 chỉ hoàn thành khi:

- [x] Baseline architecture đúng contract.
- [x] Unit tests local và Kaggle pass.
- [x] Epoch-aware augmentation được kiểm chứng.
- [x] Real batch forward pass.
- [x] One-batch overfit sanity check pass.
- [x] Pilot run pass.
- [x] Full training hoàn thành trên derived train.
- [x] Best checkpoint được chọn bằng derived validation.
- [x] Threshold được chọn chỉ bằng derived validation.
- [x] Final derived-test evaluation chạy với configuration đã khóa.
- [x] Accuracy, Precision, Recall, F1, ROC-AUC và confusion matrix được lưu.
- [x] Git commit, config, seed và environment được ghi lại.
- [x] Failure cases được định danh: 65 false positives, 39 false negatives; qualitative video review là follow-up cho báo cáo.
- [x] Không có split leakage hoặc test-label leakage đã quan sát; giới hạn duplicate/provenance được báo cáo riêng.
- [x] Không có metric giả hoặc metric tham khảo bị ghi như kết quả project.

Chỉ sau Definition of Done này mới bắt đầu Phase 3 Temporal Transformer.

## 10. Rủi ro và biện pháp

| Rủi ro | Biện pháp |
|---|---|
| Static augmentation giữa các epoch | Thêm epoch-aware deterministic RNG trước full training |
| Kaggle kernel giữ import state cũ | Install trước import; restart session khi cần |
| Resolution đa dạng và direct resize méo hình | Ghi limitation; chỉ thử letterbox như controlled follow-up |
| Backbone frozen underfit | Chỉ unfreeze có kiểm soát sau sanity/pilot evidence |
| GPU out-of-memory | Đo pilot, giảm batch size; không thay đổi frame protocol tùy tiện |
| DataLoader bottleneck | Đo throughput rồi điều chỉnh workers/pin memory |
| Test leakage | Training CLI không nhận/load test; evaluation tách riêng |
| Threshold overfit test | Chọn và lưu threshold từ derived validation trước test |
| Kaggle base-package conflict | Kiểm tra direct imports/tests; ghi caveat thay vì coi toàn bộ `pip check` là project failure |
| Mất checkpoint khi session kết thúc | Ghi vào `/kaggle/working`, Save Version và tải/attach output |

## 11. Thứ tự công việc ngay tiếp theo

1. Commit các artifact nhẹ và tài liệu Phase 2; không commit checkpoint.
2. Không chạy lại baseline test hoặc retune threshold từ test result.
3. Implement Temporal Transformer ở local với controlled ablation giữ nguyên manifest, preprocessing và evaluation protocol.
4. Chạy local tests rồi Kaggle smoke/overfit/pilot trước full Transformer training.
5. Chỉ evaluate Transformer test sau khi config/checkpoint/threshold của Transformer đã khóa bằng validation.
6. Sau khi nhánh violence chính ổn định, tiếp tục normal-profile nhẹ riêng cho Ped2 và Avenue.
