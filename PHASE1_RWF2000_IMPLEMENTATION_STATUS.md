# BÁO CÁO HIỆN TRẠNG IMPLEMENTATION

## Phase 1 — RWF-2000 Violence Data Pipeline Foundation

**Project:** Hybrid Surveillance-Video Violence and Anomaly Detection  
**Ngày tổng hợp ban đầu:** 16/09/2026

**Cập nhật gần nhất:** 06/10/2026

**Phạm vi báo cáo:** trạng thái RWF-2000 data pipeline sau real-data verification trên Kaggle

**Nguồn định hướng có hiệu lực:** Section 0 của `PROJECT_CONTEXT.md` và `AGENTS.md`

---

## 0. Trạng thái hiện hành — Phase 1 hoàn thành

> Phần 0 này là trạng thái hiện hành và thay thế các kết luận runtime cũ trong phần lịch sử bên dưới. Các mục từ phần 1 trở đi được giữ lại để bảo toàn quá trình implementation local trước khi dataset thật được attach.

### 0.1. Kết luận

**Phase 1 — RWF-2000 real-data pipeline verification: COMPLETE.**

Pipeline đã được chạy end-to-end trên RWF-2000 thật trong Kaggle Notebook:

```text
RWF-2000 thật
-> dataset layout và class mapping
-> video decoding và metadata inspection
-> deterministic manifest
-> train/validation/held-out protocol
-> RWF2000Dataset
-> 16 ordered frames
-> tensor [16, 3, 224, 224]
-> DataLoader
-> batch [B, 16, 3, 224, 224]
```

Kaggle Version #2 đã `Save & Run All` thành công từ đầu trong fresh run.

Chưa có model, training, checkpoint hoặc model metric nào được tạo.

### 0.2. Nguồn bằng chứng và phạm vi xác nhận

Các kết quả dưới đây được ghi nhận từ báo cáo Kaggle Version #2 do người dùng cung cấp ngày 06/10/2026, chạy trên Git commit:

```text
bbe275956648aa2830404deb23b723e93c1ea29c
```

Các artifact hiện nằm trên Kaggle, chưa được đưa vào repository local:

- `/kaggle/working/rwf2000_phase1/inspection.json`
- `/kaggle/working/rwf2000_phase1/rwf2000_manifest.csv`
- `/kaggle/working/rwf2000_phase1/rwf2000_manifest.json`

Repository không lưu dataset hoặc các artifact lớn. Báo cáo này ghi nhận kết quả của run Kaggle; không tuyên bố đã tự tái chạy dataset thật ở local.

### 0.3. Môi trường Kaggle đã đo

| Thành phần | Giá trị |
|---|---|
| Platform | Linux 6.18.48+, x86_64, glibc 2.39 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cpu |
| OpenCV | 4.14.0 |
| Accelerator | None |
| CUDA | Không khả dụng, đúng với Phase 1 CPU-only |

Repository được clone vào `/kaggle/working/Video-anomaly-detection`. Editable install `pip install -e ".[test]"` thành công, package import được và CLI có hai subcommand `inspect`/`manifest`.

`pip check` không sạch vì conflict giữa các package có sẵn trong Kaggle base image (`bigframes`, `google-colab`, `dopamine-rl`, `moviepy`). Đây là caveat của môi trường Kaggle, không phải lỗi đã quan sát từ dependency trực tiếp của `rwf2000-pipeline`. Bằng chứng hỗ trợ gồm package import thành công, toàn bộ unit tests pass, CLI hoạt động và real-data pipeline pass.

### 0.4. Dataset layout đã đo

Dataset được attach từ Kaggle input của owner Dania Zehra:

```text
/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000
├── train
│   ├── Fight
│   └── NonFight
└── val
    ├── Fight
    └── NonFight
```

Protocol đã khóa cho bản dataset này:

```text
source train
-> derived train + derived val

source val
-> derived test (held-out, không dùng để tuning)
```

Tên thư mục nguồn `val` không được hiểu là tuning validation trong project. Nó được giữ nguyên làm held-out same-domain evaluation partition.

### 0.5. Kết quả inspection thật

| Source split | Non-violence | Violence | Tổng | Short clips | Decode error/mismatch |
|---|---:|---:|---:|---:|---:|
| `train` | 800 | 800 | 1.600 | 0 | 0 |
| `val` (held-out) | 200 | 200 | 400 | 0 | 0 |
| **Tổng** | **1.000** | **1.000** | **2.000** | **0** | **0** |

Tất cả 2.000 clip được báo cáo ở 30 FPS. Resolution đa dạng; các resolution phổ biến gồm 320x240, 1280x720, 640x360, 480x360 và một số video 1920x1080.

Kết luận inspection:

- Hai class cân bằng trong cả source training và held-out partition.
- Không có clip dưới 16 decoded frames.
- Không có video open/decode failure được quan sát.
- Không có metadata/decode frame-count mismatch được quan sát.
- Sự đa dạng resolution được ghi nhận; baseline ban đầu vẫn giữ direct resize 224x224 để không thay đổi protocol sau khi nhìn test, còn letterbox/aspect-ratio preservation chỉ được xem là một thay đổi có kiểm soát sau baseline.

### 0.6. Manifest đã khóa

Manifest configuration:

| Trường | Giá trị |
|---|---|
| Schema | `rwf2000-manifest-v1` |
| Seed | `42` |
| Validation fraction | `0.2` |
| Non-violence | `0` |
| Violence | `1` |

Derived assignments:

| Derived split | Non-violence | Violence | Tổng |
|---|---:|---:|---:|
| `train` | 640 | 640 | 1.280 |
| `val` | 160 | 160 | 320 |
| `test` | 200 | 200 | 400 |
| **Tổng** | **1.000** | **1.000** | **2.000** |

Protocol check đã pass:

- Derived validation chỉ lấy từ source `train`.
- Toàn bộ source `val` được giữ trong derived `test`.
- Filesystem order không quyết định assignment.
- Seed, validation fraction, schema và label mapping đã được lưu.

Phạm vi phát biểu leakage phải được giữ chính xác: manifest đã bảo vệ train/validation/held-out assignment ở cấp path/clip theo implementation hiện tại. Source-video grouping, content duplicate, perceptual duplicate và provenance chưa được xác minh vì dataset không cung cấp metadata tương ứng và pipeline chưa chạy content hashing.

### 0.7. Runtime verification

Kết quả unit tests trên Kaggle:

```text
17 passed in 2.12s
```

Real sample đã kiểm tra:

| Trường | Kết quả |
|---|---|
| Dataset records trong derived train | 1.280 |
| Sample path | `train/Fight/-1l5631l3fg_0.avi` |
| Label | `1.0` / Violence |
| Tensor shape | `[16, 3, 224, 224]` |
| Tensor dtype | `torch.float32` |
| Finite tensor | Pass |
| Ordered indices | `[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149]` |

Real DataLoader batch:

| Trường | Kết quả |
|---|---|
| Batch shape | `[2, 16, 3, 224, 224]` |
| Tensor dtype | `torch.float32` |
| Labels | `[1.0, 1.0]` |
| Finite tensor | Pass |

Batch có hai nhãn giống nhau không phải lỗi vì smoke test dùng `shuffle=False` và chỉ kiểm tra wiring/contract, không kiểm tra class balance từng batch.

Trong interactive kernel, editable install ban đầu chưa được nhận ngay nên `REPO_DIR / "src"` đã được thêm vào `sys.path`. Package vẫn import được trong Python process mới. Với notebook tiếp theo, nên cài package trước mọi import và restart kernel/session khi cần, thay vì coi `sys.path` workaround là behavior của core package.

### 0.8. Artifacts và Save & Run All

| Artifact | Kích thước báo cáo |
|---|---:|
| `inspection.json` | 794,14 KB |
| `rwf2000_manifest.csv` | 363,21 KB |
| `rwf2000_manifest.json` | 0,60 KB |

Version #1 thất bại vì notebook coi `pip check` exit code 1 của Kaggle base image là fatal. Cell sau đó được điều chỉnh để ghi nhận caveat mà không che lỗi install/import/test của project.

Kaggle từng báo `ConcurrencyViolation` và `Sequence number must match Draft record`; đây là lỗi đồng bộ notebook. Sau refresh và chạy lại, Version #2 hoàn thành thành công và giữ đủ artifacts.

### 0.9. Những gì chưa thực hiện

- Chưa implement `ResNet18 + temporal average pooling`.
- Chưa có training loop, optimizer, scheduler hoặc checkpoint.
- Chưa chạy one-batch overfit sanity check.
- Chưa có Accuracy, Precision, Recall, F1, ROC-AUC hoặc confusion matrix của project.
- Chưa chọn threshold bằng derived validation.
- Chưa đánh giá derived test.
- Chưa implement Temporal Transformer.
- Chưa implement lightweight anomaly branch cho Ped2/Avenue.
- Chưa kiểm tra source-video provenance hoặc content duplicates.

### 0.10. Quyết định chuyển phase

Không còn blocker data-pipeline trước baseline. Bước tiếp theo là Phase 2:

```text
16 RGB frames
-> pretrained ResNet18 per frame
-> [B, T, 512] frame features
-> temporal average pooling
-> [B, 512]
-> linear classifier
-> one violence logit per clip
```

Kế hoạch chi tiết được lưu trong `PHASE2_RWF2000_BASELINE_PLAN.md`.

---

# LỊCH SỬ IMPLEMENTATION LOCAL TRƯỚC KAGGLE

Các phần bên dưới phản ánh snapshot ngày 16/09/2026. Những câu như “pytest chưa chạy”, “dependency chưa cài” hoặc “RWF-2000 thật chưa được inspect” chỉ đúng tại thời điểm lịch sử đó và đã được thay thế bởi Section 0 ở trên.

---

## 1. Tóm tắt điều hành

Iteration hiện tại đã tạo xong nền tảng mã nguồn ban đầu cho nhánh phát hiện bạo lực bằng RWF-2000. Phần đã xây gồm:

- Package Python tối thiểu theo `src` layout.
- Quy tắc ánh xạ nhãn `Violence` / `Non-violence` tường minh.
- Sampler lấy đúng 16 frame theo thứ tự thời gian và không vượt clip boundary.
- Video decoder tuần tự dùng OpenCV, có thể thay bằng decoder giả trong test.
- Preprocessing chung cho train/evaluation/inference.
- Công cụ kiểm tra cấu trúc và metadata dataset.
- Nền tảng tạo split/manifest có seed và có khả năng tái lập.
- `Dataset` và `DataLoader` foundation theo tensor contract đã chốt.
- Unit-test source cho label mapping, temporal order, clip boundary, manifest determinism, inspection và DataLoader smoke test.
- `pyproject.toml` chứa dependency và cấu hình test tối thiểu.

Iteration không triển khai model, không train, không tải dataset, không tạo metric và không đụng vào nhánh anomaly Ped2/Avenue.

Điểm quan trọng về định hướng:

- Project vẫn là **hybrid surveillance-video detection project**.
- Nhánh violence bằng RWF-2000 là bắt buộc.
- Nhánh lightweight anomaly cho Ped2 và Avenue vẫn là bắt buộc trong sản phẩm cuối.
- Việc iteration này chỉ tập trung RWF-2000 **không có nghĩa là loại bỏ nhánh anomaly**.
- Temporal Transformer, baseline ResNet18 và toàn bộ phần model chỉ được làm sau khi data pipeline đã được xác minh đầy đủ hơn.

Trạng thái xác minh hiện tại:

- Kiểm tra cú pháp toàn bộ source và tests: **pass**.
- Các kiểm tra thủ công không cần dependency ngoài cho label, sampler, manifest và inspection: **pass**.
- CLI help và xử lý dataset path không tồn tại: **pass**.
- Bộ `pytest` hoàn chỉnh: **chưa chạy được** vì runtime hiện tại chưa cài `pytest`, `torch`, `numpy`, `opencv-python`.
- DataLoader smoke test: đã viết nhưng **chưa thực thi được** do thiếu PyTorch.
- Decode video thật và kiểm tra RWF-2000 thật: **chưa thể thực hiện** vì chưa có dataset path và chưa có OpenCV trong runtime.
- Không có kết quả metric mô hình nào; không có số liệu nào được trình bày như kết quả thực nghiệm.

---

## 2. Phạm vi và nguyên tắc đã tuân thủ

### 2.1. Phạm vi được phép trong iteration

- Tạo source skeleton tối thiểu.
- Tạo dependency/test configuration tối thiểu.
- Tạo utility kiểm tra RWF-2000.
- Tạo split/manifest foundation.
- Tạo sampler 16 frame.
- Tạo preprocessing foundation.
- Tạo Dataset/DataLoader foundation.
- Viết test bằng synthetic/minimal temporary data.
- Thực hiện các smoke check có thể chạy trong môi trường local hiện tại.

### 2.2. Phần chủ động không làm

- Không implement ResNet18 + temporal average pooling.
- Không implement Temporal Transformer.
- Không implement loss, optimizer, training loop hoặc checkpointing.
- Không train full dataset hoặc một phần dataset.
- Không implement Ped2/Avenue lightweight anomaly branch.
- Không implement ConvLSTM.
- Không implement demo UI.
- Không tải RWF-2000 hoặc dataset khác.
- Không giả định RWF-2000 đã có trên máy.
- Không tạo accuracy, F1, ROC-AUC, confusion matrix hoặc metric giả.
- Không tune threshold.
- Không dùng test labels để chọn threshold hoặc hyperparameter.
- Không commit thay người dùng.

### 2.3. Nguyên tắc experimental integrity được giữ

- Official test được coi là partition bất biến.
- Validation chỉ được tách từ official train.
- Split diễn ra ở cấp clip, không ở cấp frame/window.
- Temporal frame order được giữ nguyên.
- Một sample không được nối frame từ nhiều clip.
- Synthetic fixtures chỉ kiểm tra infrastructure, không phải bằng chứng về RWF-2000.
- Target/estimate trong `PROJECT_CONTEXT.md` không được chuyển thành measured result.
- Không tuyên bố source-level leakage đã được loại bỏ khi chưa có source-video metadata thật.

---

## 3. Quy trình multi-agent đã thực hiện

Theo yêu cầu, quá trình bắt đầu bằng ba subagent read-only chạy song song. Không agent nào sửa file trong bước khảo sát.

### 3.1. `project_explorer`

Nhiệm vụ:

- Đọc `AGENTS.md` và Section 0 của `PROJECT_CONTEXT.md`.
- Kiểm tra trạng thái repository thực tế.
- Phân biệt code đã implement với kế hoạch/tài liệu lịch sử.
- Đề xuất source skeleton tối thiểu cho Phase 1.

Kết luận chính:

- Repository ban đầu chỉ có tài liệu, presentation và cấu hình agents.
- Không có `src/`, `tests/`, config Python, Dataset/DataLoader hoặc model.
- Không có RWF-2000 trong repository.
- Các cấu trúc source trong phần lịch sử của `PROJECT_CONTEXT.md` chỉ là kế hoạch, không phải implementation.
- Section 0 phải thắng mọi phần Ped2-only lịch sử nếu có xung đột.

### 3.2. `ml_reviewer`

Nhiệm vụ:

- Rà soát contract ML của RWF-2000 pipeline.
- Tập trung vào label mapping, frame order, clip boundary, 16-frame sampling, preprocessing, augmentation, tensor shape, train/inference consistency và leakage.

Kết luận chính:

- Không được dùng substring để nhận diện nhãn vì `NonFight` chứa chuỗi `Fight`.
- Canonical mapping nên là `Non-violence = 0`, `Violence = 1`.
- Sampler phải dùng interval rõ ràng và không thể vượt clip boundary.
- Frame phải được decode tuần tự hoặc theo index có kiểm soát.
- Augmentation spatial/appearance phải dùng cùng tham số cho toàn bộ 16 frame trong một clip.
- Eval/inference phải deterministic và không dùng random augmentation.
- Tensor sample nên là `[T, C, H, W]`; batch là `[B, T, C, H, W]`.
- Không được tuyên bố đã chống source-video leakage nếu chưa có metadata nguồn.

### 3.3. `experiment_reviewer`

Nhiệm vụ:

- Rà soát train/validation/test protocol.
- Đề xuất manifest/split có khả năng tái lập.
- Rà soát seed và nguy cơ data leakage.
- Bảo vệ test set khỏi tuning.

Kết luận chính:

- Nếu dữ liệu có official train/test thì giữ test bất biến.
- Validation phải được tách từ official train, stratified theo class và ở cấp clip.
- Filesystem enumeration order không được ảnh hưởng split.
- Input phải được sort ổn định trước khi shuffle bằng local seeded RNG.
- Manifest phải lưu seed, split rule, class mapping và schema version.
- Test, same-domain test và external test phải được báo cáo riêng.

### 3.4. `implementation_worker`

Sau khi main agent tổng hợp và tự đối chiếu Section 0, một worker duy nhất được giao quyền sửa code trong bounded scope.

Worker đã:

- Tạo package `src/rwf2000`.
- Tạo tests.
- Tạo `pyproject.toml`.
- Chạy compile/manual checks có thể chạy.
- Không commit.
- Không tải dependency hoặc dataset.

Sau vòng review chính, worker được yêu cầu sửa ba điểm nhỏ:

- Không cho DataLoader smoke test tự động skip khi thiếu dependency bắt buộc.
- Sửa wording metadata để không phủ nhận chính các count quan sát từ supplied directory.
- Thêm test cho inspection utility.

---

## 4. Trạng thái repository hiện tại

### 4.1. Git

Tại thời điểm tổng hợp:

- Branch hiện tại: `main`.
- Repository chưa có commit ban đầu.
- Các file hiện có đều đang ở trạng thái untracked.
- Không có commit nào được tạo trong iteration.
- Không có dataset, checkpoint, credential hoặc `.env` được thêm.

Do repository chưa có commit baseline, `git diff` không thể hiện thay đổi theo cách thông thường. Trạng thái được kiểm tra bằng `git status --short --untracked-files=all` và đối chiếu danh sách file trước/sau iteration.

### 4.2. File tree liên quan trực tiếp đến implementation

```text
pyproject.toml
src/
└── rwf2000/
    ├── __init__.py
    ├── __main__.py
    ├── dataset.py
    ├── decoder.py
    ├── inspection.py
    ├── labels.py
    ├── manifest.py
    ├── preprocessing.py
    └── sampling.py
tests/
├── README.md
├── conftest.py
├── test_dataset.py
├── test_inspection.py
├── test_labels.py
├── test_manifest.py
└── test_sampling.py
```

### 4.3. Runtime local hiện tại

Runtime đã kiểm tra:

```text
Python 3.14.6
pytest: chưa cài
torch: chưa cài
numpy: chưa cài
cv2/OpenCV: chưa cài
```

`pyproject.toml` đã khai báo:

```text
Python >= 3.10
numpy >= 1.24
torch >= 2.0
opencv-python >= 4.8
pytest >= 7  (test extra)
```

Chưa có package nào được cài trong iteration vì việc tải/cài dependency không được tự ý thực hiện.

---

## 5. Mô tả chi tiết implementation hiện có

## 5.1. `pyproject.toml`

Vai trò:

- Định nghĩa package `rwf2000-pipeline` phiên bản `0.1.0`.
- Dùng setuptools và `src` layout.
- Khai báo dependency runtime tối thiểu.
- Khai báo `pytest` trong optional dependency `test`.
- Chỉ định `tests/` là test path.

Không có dependency cho model zoo, training framework cao cấp, tracking hoặc UI vì các phần đó nằm ngoài scope.

## 5.2. `src/rwf2000/labels.py`

Canonical mapping:

```text
Non-violence -> 0
Violence     -> 1
```

Alias được hỗ trợ bằng exact normalized matching:

| Input directory | Canonical name | Label |
|---|---|---:|
| `NonFight` | `Non-violence` | 0 |
| `NonViolence` | `Non-violence` | 0 |
| `Non-violence` | `Non-violence` | 0 |
| `Fight` | `Violence` | 1 |
| `Violence` | `Violence` | 1 |

Tên được `strip()` và `casefold()` trước khi lookup, nhưng toàn bộ normalized string phải khớp alias.

Ví dụ:

- `NonFight` hợp lệ và trả về 0.
- `nonfight` hợp lệ và trả về 0.
- `Fight` hợp lệ và trả về 1.
- `NonFighting` bị từ chối.
- `FightExtra` bị từ chối.
- Không có logic kiểu `"fight" in class_name`.

Unknown label directory phát sinh `UnknownClassError` với thông báo liệt kê alias được hỗ trợ.

## 5.3. `src/rwf2000/sampling.py`

API chính:

```python
sample_uniform_indices(
    num_frames,
    num_samples=16,
    start=0,
    stop=None,
)
```

Contract:

- Sample từ interval half-open `[start, stop)`.
- Mặc định sample toàn clip `[0, num_frames)`.
- Trả đúng `num_samples`, mặc định là 16.
- Index đầu là `start`.
- Index cuối là `stop - 1`.
- Mọi index đều tăng dần nghiêm ngặt khi interval có ít nhất `num_samples` frame.
- Không dùng random sampling.
- Không dùng floating-point rounding; index được tính bằng integer arithmetic.
- Kết quả deterministic.
- Không thể trả index ngoài clip/window.

Short-clip policy hiện tại:

- `strict error`.
- Nếu interval có ít hơn 16 frame thì phát sinh `ValueError`.
- Chưa có repeat-last-frame padding.
- Chưa có validity mask.
- Tuyệt đối không mượn frame từ clip khác.

## 5.4. `src/rwf2000/decoder.py`

`OpenCVDecoder` là decoder mặc định cho video thật.

Hành vi:

- Open video bằng `cv2.VideoCapture`.
- Đọc frame tuần tự từ index 0 đến index lớn nhất được yêu cầu.
- Chỉ thu các frame có index nằm trong danh sách sampler.
- Yêu cầu danh sách index phải unique và tăng dần.
- Chuyển frame từ BGR của OpenCV sang RGB.
- Giải phóng `VideoCapture` trong `finally`.
- Báo lỗi rõ nếu video không mở được hoặc decode thất bại trước frame yêu cầu.

Lý do đọc tuần tự:

- Tránh rủi ro random seek trả sai vị trí với một số codec.
- Bảo toàn temporal order rõ ràng.

Trade-off hiện tại:

- Với video dài, đọc tuần tự từ đầu đến frame lớn nhất có thể chậm hơn seek.
- RWF-2000 dự kiến dùng clip ngắn, nhưng điều này chưa được đo trên dataset thật trong repository.
- Tối ưu decoder chưa nằm trong Phase 1.

Decoder có thể được inject vào Dataset. Cơ chế này cho phép test logic DataLoader bằng synthetic decoder mà không cần codec hoặc video thật.

## 5.5. `src/rwf2000/preprocessing.py`

`PreprocessConfig` mặc định:

```text
height = 224
width = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
horizontal_flip_prob = 0.5
brightness_delta = 0.1
contrast_range = (0.9, 1.1)
crop_scale = (0.9, 1.0)
```

Base preprocessing:

1. Nhận RGB frames dạng sequence `[T, H, W, C]` hoặc tensor tương thích.
2. Chuyển sang `[T, C, H, W]`.
3. Chuyển sang `float32`.
4. Scale `uint8 [0,255]` về `[0,1]`.
5. Resize trực tiếp về kích thước cấu hình, mặc định `224 × 224`.
6. Chuẩn hóa bằng ImageNet mean/std.

Validation hiện có:

- Frame tensor phải có bốn chiều.
- Phải có đúng ba RGB channels.
- Không chấp nhận sequence rỗng.
- Không chấp nhận NaN/Inf.
- Không chấp nhận giá trị ngoài `[0,255]` trước scaling.
- Output luôn là `float32`.

Train-only augmentation:

- Horizontal flip.
- Brightness shift nhẹ.
- Contrast scaling nhẹ.
- Controlled random crop.

Mọi tham số augmentation được sample một lần cho cả clip, sau đó áp dụng giống nhau lên toàn bộ frame. Điều này tránh tạo flicker hoặc chuyển động giả do mỗi frame nhận transform khác nhau.

Eval/inference:

- Không dùng random augmentation.
- Dùng cùng base resize và normalization với train.
- Không đảo frame, shuffle frame hoặc temporal reversal.

Giới hạn hiện tại:

- Resize đang là direct resize, có thể làm thay đổi aspect ratio.
- Letterbox/padding giữ tỷ lệ chưa được implement; Section 0 xem đây là phương án cần kiểm tra sau.
- Dataset tạo RNG từ seed và clip ID nên augmentation của một clip có tính tái lập cao, nhưng chưa có epoch-aware augmentation schedule. Nếu dùng nguyên trạng cho training, cùng clip có thể nhận cùng transform giữa các lần truy cập khi seed không đổi.

## 5.6. `src/rwf2000/manifest.py`

### Mục đích

- Discover clip từ layout đã chỉ định.
- Giữ official test bất biến.
- Tách validation từ official train.
- Lưu assignment vào CSV deterministic.
- Lưu metadata sidecar JSON deterministic.

### Video extension mặc định

```text
.avi, .mkv, .mov, .mp4, .mpeg, .mpg, .webm
```

### Manifest schema hiện tại

Mỗi record gồm:

| Field | Ý nghĩa |
|---|---|
| `schema_version` | Phiên bản schema, hiện là `rwf2000-manifest-v1` |
| `seed` | Seed dùng để tạo validation assignment |
| `val_fraction` | Tỷ lệ validation được yêu cầu |
| `label_map` | Canonical label map serialized dạng JSON |
| `relative_path` | Clip path tương đối so với dataset root |
| `clip_id` | SHA-256 của relative path |
| `source_split` | Tên split directory nguồn, ví dụ `train` hoặc `test` |
| `derived_split` | Split dùng bởi pipeline: `train`, `val` hoặc `test` |
| `label` | 0 hoặc 1 |
| `label_name` | `Non-violence` hoặc `Violence` |

Lưu ý:

- `clip_id` là identity ổn định theo path, không phải content hash.
- Di chuyển/đổi tên file sẽ làm thay đổi `clip_id`.
- Hai file khác path nhưng cùng nội dung chưa được phát hiện bằng content hash.

### Split algorithm

1. Yêu cầu hai split directory nguồn tường minh: train và test.
2. Sort file theo relative path đã `casefold()`.
3. Gom official-train clip theo label.
4. Dùng một local `random.Random(seed)` để shuffle từng class.
5. Chọn validation count theo class.
6. Gán phần còn lại vào derived train.
7. Gán mọi official-test clip vào derived test mà không resplit.
8. Sort assignment theo relative path trước khi ghi CSV.

Validation count:

- Nếu class có dưới hai clip hoặc `val_fraction = 0`, validation count là 0.
- Với `val_fraction > 0` và class có ít nhất hai clip, giữ ít nhất một validation clip.
- Luôn giữ ít nhất một training clip cho class đó.

### Determinism

Cùng dataset layout, seed và configuration sẽ tạo:

- CSV byte-for-byte giống nhau.
- JSON metadata byte-for-byte giống nhau.
- Cùng `clip_id` và split assignment.

### Sidecar metadata

JSON sidecar chứa:

- Schema version.
- Seed.
- Validation fraction.
- Label map.
- Tên source train/test directory.
- Tổng số record.
- Count theo derived split và canonical class.
- Limitations mô tả phạm vi xác minh.

### Overlap/leakage protection hiện có

- Reject duplicate relative path.
- Dùng class canonical + path bên dưới class directory như một heuristic để phát hiện tên/path clip lặp giữa source splits.
- Reject duplicate clip ID khi load manifest.
- Bảo đảm train/validation assignment diễn ra ở cấp clip.

Chưa có:

- Content hash cho video.
- `source_video_id` thật.
- Group-aware splitting theo source video.
- Perceptual duplicate detection.
- Xác minh rằng hai clip khác tên có cùng nguồn quay.

Vì vậy pipeline **chưa thể tuyên bố source-level leakage-free** trên RWF-2000 thật.

## 5.7. `src/rwf2000/inspection.py`

API chính:

```python
inspect_dataset(
    dataset_root,
    splits=("train", "test"),
    video_extensions=None,
    short_clip_length=16,
)
```

Công cụ:

- Yêu cầu dataset root tường minh.
- Không tải dataset.
- Không tự suy luận thư mục `val`.
- Yêu cầu các split name không trùng nhau.
- Reject missing root hoặc missing split directory bằng lỗi rõ ràng.
- Reject unknown class directory thông qua label mapping trung tâm.

Khi OpenCV có sẵn, inspection đọc:

- Reported frame count.
- Decoded frame count.
- FPS.
- Width.
- Height.
- Duration tính từ decoded frame count/FPS.
- Short-clip flag.
- Decode error hoặc mismatch giữa metadata count và decoded count.

Khi OpenCV không có:

- Vẫn thống kê được số clip theo split và label từ filesystem.
- Thêm limitation giải thích rằng video metadata/decode chưa được kiểm tra.

Report luôn ghi rõ:

- Nó chỉ mô tả supplied directory.
- Nó không chứng minh supplied directory là official RWF-2000 release.
- Nó không thể xác minh source-level duplicate/leakage nếu thiếu provenance metadata.

## 5.8. `src/rwf2000/dataset.py`

### `RWF2000Dataset`

Input:

- Dataset root.
- Manifest CSV hoặc sequence `ManifestRecord`.
- Optional derived split filter.
- Optional injected decoder.
- `PreprocessConfig`.
- Training/evaluation mode.
- Seed.
- Số frame, mặc định 16.

Path safety:

- Resolve dataset root và clip path.
- Reject relative path thoát ra ngoài dataset root.
- Reject clip path không tồn tại.

Per-sample flow:

```text
Manifest record
-> resolve clip path
-> lấy frame count
-> sample 16 ordered indices
-> decode đúng các indices
-> clip-consistent preprocessing
-> trả frames + label + metadata
```

Sample dictionary:

```text
frames: float32 tensor [16, 3, H, W]
label:  float32 scalar tensor
metadata:
  clip_id
  relative_path
  source_split
  derived_split
  label
  label_name
  frame_indices
```

DataLoader batch contract:

```text
frames: [B, 16, 3, H, W]
label:  [B]
```

Label dùng float32 để phù hợp với future single-logit `BCEWithLogitsLoss` contract. Model/loss chưa được implement trong iteration này.

### `make_dataloader`

Hỗ trợ:

- `batch_size`.
- `shuffle`.
- `num_workers`.
- `seed`.
- `drop_last`.
- PyTorch `Generator` có seed.
- `worker_init_fn` seed Python và NumPy worker RNG từ PyTorch worker seed.

Tests dự kiến dùng `num_workers=0` để giảm nondeterminism và phụ thuộc môi trường.

## 5.9. `src/rwf2000/__main__.py`

CLI có hai subcommand.

### Inspection

```powershell
python -m rwf2000 inspect `
  --root D:\path\to\RWF-2000 `
  --splits train test
```

Ghi JSON report ra file:

```powershell
python -m rwf2000 inspect `
  --root D:\path\to\RWF-2000 `
  --splits train test `
  --output artifacts\rwf2000_inspection.json
```

### Manifest

```powershell
python -m rwf2000 manifest `
  --root D:\path\to\RWF-2000 `
  --output artifacts\rwf2000_manifest.csv `
  --train-split train `
  --test-split test `
  --val-fraction 0.2 `
  --seed 42
```

Nếu một bản phân phối dùng thư mục `val` làm held-out official test, phải ánh xạ tường minh:

```powershell
python -m rwf2000 manifest `
  --root D:\path\to\RWF-2000 `
  --output artifacts\rwf2000_manifest.csv `
  --train-split train `
  --test-split val
```

Tool không tự quyết định `val` là validation tuning set hay official held-out set. Quyết định này phải dựa trên provenance/layout của dataset thật.

---

## 6. Dataset layout được code hỗ trợ

Layout mặc định:

```text
RWF_ROOT/
├── train/
│   ├── Fight/
│   │   └── *.avi|*.mp4|...
│   └── NonFight/
│       └── *.avi|*.mp4|...
└── test/
    ├── Fight/
    │   └── *.avi|*.mp4|...
    └── NonFight/
        └── *.avi|*.mp4|...
```

Canonical aliases khác cũng được chấp nhận, ví dụ `Violence` và `Non-violence`.

Các giả định không được code tự đưa ra:

- Không giả định dataset root nằm trong repository.
- Không giả định split directory luôn tên `train/test` nếu người dùng truyền tên khác.
- Không tự đổi `val` thành test hoặc validation.
- Không giả định mọi video decode được.
- Không giả định mọi clip có ít nhất 16 frame.
- Không giả định class balance đúng như tài liệu.
- Không giả định source-video IDs có sẵn.

---

## 7. Test infrastructure đã tạo

## 7.1. `tests/test_labels.py`

Kiểm tra:

- Các alias non-violence map về 0.
- Các alias violence map về 1.
- `NonFight` không bị nhầm thành `Fight`.
- Unknown/extended class name bị reject.

## 7.2. `tests/test_sampling.py`

Kiểm tra:

- Trả đúng 16 indices.
- Temporal order tăng dần.
- Sample đầu/cuối khớp interval endpoints.
- Mọi index nằm trong `[start, stop)`.
- Clip/window ngắn hơn 16 frame bị reject.
- Không pad hoặc vượt boundary.

## 7.3. `tests/test_manifest.py`

Dùng temporary synthetic file tree để kiểm tra:

- Cùng input + seed tạo CSV byte-identical.
- JSON sidecar byte-identical.
- `clip_id` assignment giống nhau.
- Train và validation không overlap.
- Official test vẫn là derived test.
- Potential same-path clip overlap giữa source splits bị reject.

Các `.mp4` trong test này chỉ là placeholder byte files. Test không coi chúng là video thật và không decode chúng.

## 7.4. `tests/test_inspection.py`

Kiểm tra:

- Missing dataset root trả lỗi rõ ràng.
- Count theo split và canonical label đúng.
- Test count độc lập với việc OpenCV có hoặc không có trong environment.

Synthetic fixture được ghi rõ không phải bằng chứng về RWF-2000.

## 7.5. `tests/test_dataset.py`

Tạo injected `SyntheticDecoder` để tránh phụ thuộc codec/video thật.

Smoke test dự kiến kiểm tra:

- DataLoader batch shape là `[2, 16, 3, 8, 8]` trong test configuration.
- Frame dtype là `torch.float32`.
- Label dtype là `torch.float32`.
- Label mapping trong batch là đúng.
- Requested frame indices tăng dần.
- Không index nào vượt frame count của clip giả.

Test import `torch` trực tiếp. Vì PyTorch là dependency bắt buộc, test sẽ fail ở environment cấu hình sai thay vì bị silently skipped.

## 7.6. Số test source hiện có

Theo định nghĩa hiện tại:

- 10 label cases sau khi mở rộng các `pytest.mark.parametrize`.
- 2 sampling cases.
- 2 manifest cases.
- 2 inspection cases.
- 1 DataLoader smoke case.

Tổng dự kiến: **17 pytest cases** khi dependency đầy đủ và collection thành công.

Con số này mô tả test definitions, không phải số test đã chạy pass.

---

## 8. Kết quả kiểm tra đã thực sự chạy

## 8.1. Compile check

Command:

```powershell
python -m compileall -q src tests
```

Kết quả:

```text
PASS
exit code: 0
```

Điều này xác nhận source/test files parse được trong Python hiện tại. Nó không xác nhận runtime behavior của PyTorch/OpenCV code.

## 8.2. Manual standard-library checks

Đã chạy một harness tạm thời dùng standard library và source package để kiểm tra:

- `NonFight -> 0`.
- `Fight -> 1`.
- Unknown class bị reject.
- Sampler trả đúng 16 ordered indices trong half-open boundary.
- Short clip bị reject.
- Hai lần build manifest với cùng input/seed tạo CSV giống từng byte.
- Hai JSON sidecar giống từng byte.
- `load_manifest()` đọc lại đúng records.
- Train/val/test disjoint theo manifest.
- Source test luôn map thành derived test.
- Inspection counts đúng trên synthetic filesystem fixture.
- Metadata JSON giữ đúng seed và validation fraction.

Kết quả:

```text
PASS
exit code: 0
```

## 8.3. CLI checks

CLI help:

```powershell
python -m rwf2000 --help
```

Kết quả:

```text
PASS
exit code: 0
```

Missing dataset root:

```powershell
python -m rwf2000 inspect --root .\__missing_rwf2000__
```

Kết quả mong đợi và quan sát được:

```text
error: RWF-2000 dataset root does not exist
exit code: 2
```

Đây được coi là pass cho error-path behavior.

## 8.4. Pytest

Command đã thử:

```powershell
python -m pytest -q
```

Kết quả:

```text
Không chạy được test collection.
Lý do: No module named pytest
exit code: 1
```

Đây là environment/dependency failure, không phải assertion failure của code.

Vì test runner không tồn tại, hiện chưa có số lượng pytest pass/fail thực tế.

## 8.5. DataLoader/OpenCV checks

Chưa chạy được:

- DataLoader smoke test vì thiếu `torch` và `numpy`.
- Preprocessing runtime test vì thiếu `torch`.
- OpenCV real decode vì thiếu `opencv-python`.
- Metadata inspection trên video thật vì thiếu OpenCV và dataset.

---

## 9. Những gì chưa làm được hoặc chưa thể xác minh

## 9.1. Thiếu dataset thật

Repository không chứa RWF-2000 và chưa nhận được dataset path từ người dùng.

Do đó chưa thể xác minh:

- Layout thực tế của bản dataset sẽ dùng.
- Split directory tên `test`, `val` hay tên khác.
- Tên class directory thực tế.
- Số lượng clip quan sát được.
- Class balance quan sát được.
- Codec và container.
- FPS.
- Resolution.
- Duration.
- Frame count thực tế.
- Clip ngắn hơn 16 frame.
- File corrupt hoặc decode mismatch.
- Source-video IDs.
- Duplicate clip/content giữa splits.
- Manifest trên full RWF-2000.
- DataLoader trên RWF-2000 thật.

Mọi con số RWF-2000 trong `PROJECT_CONTEXT.md` vẫn là source-backed project context hoặc mục tiêu kế hoạch, không phải đo đạc từ repository này.

## 9.2. Thiếu dependency runtime

Chưa cài:

- `pytest`.
- `torch`.
- `numpy`.
- `opencv-python`.

Vì vậy chưa thể tuyên bố full test suite pass.

## 9.3. Chưa xác minh train-time behavior dài hạn

Chưa có training loop nên chưa xác minh:

- DataLoader multi-worker behavior trên Windows.
- Augmentation variation giữa epochs.
- Throughput/FPS của decoder.
- Memory consumption với batch size thực tế.
- Pinned memory hoặc persistent workers.
- Reproducibility giữa CPU/GPU hosts.
- Reproducibility khi thay package versions.

## 9.4. Chưa có source-group-aware split

Manifest split hiện là stratified clip-level split. Nếu RWF-2000 cung cấp source-video identity hoặc nhiều clip từ cùng source video, cần thêm group-aware split để tránh related clips rơi vào train và validation.

Hiện tại không có đủ metadata để implement hoặc xác minh phần này một cách trung thực.

## 9.5. Chưa có content hashing

`clip_id` hiện hash relative path, không hash video bytes.

Hệ quả:

- Không phát hiện được hai file khác tên nhưng cùng nội dung.
- Không phát hiện near-duplicate/re-encoded video.
- Không chứng minh được no-content-overlap giữa split.

## 9.6. Chưa có model hoặc metric

Chưa có:

- ResNet18 backbone.
- Temporal average pooling baseline.
- Temporal Transformer.
- Binary classification head.
- BCE/BCEWithLogits loss wiring.
- Optimizer.
- Training/validation loop.
- Checkpoint selection.
- Threshold tuning.
- Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
- Inference pipeline.
- Demo.

Không có metric nào cần hoặc được phép báo cáo ở trạng thái hiện tại.

## 9.7. Nhánh anomaly chưa triển khai

Chưa có:

- Ped2 Dataset/DataLoader.
- Avenue Dataset/DataLoader.
- Pretrained embedding normal profile.
- Anomaly score.
- Per-dataset normal memory.
- AUROC/AP evaluation.
- Fusion rule giữa violence probability và anomaly score.

Đây là phần bắt buộc trong sản phẩm cuối nhưng nằm ngoài Phase 1 hiện tại. Nó không bị loại khỏi roadmap.

---

## 10. Rủi ro kỹ thuật hiện tại

| Rủi ro | Trạng thái | Cách xử lý tiếp theo |
|---|---|---|
| Dataset distribution dùng `val` làm official test | Chưa biết | Kiểm tra provenance/layout và truyền `--test-split val` tường minh nếu đúng |
| `NonFight` bị đảo nhãn | Đã có protection và test source | Chạy full pytest sau khi cài dependency |
| Frame order sai | Sampler/decoder đã thiết kế bảo toàn | Chạy test và kiểm tra trên video có frame index trực quan |
| Clip ngắn hơn 16 frame | Strict error | Inspection dataset thật trước khi training; quyết định padding policy nếu thực sự cần |
| Random augmentation gây flicker | Đã dùng clip-consistent parameters | Thêm runtime tensor test khi PyTorch có sẵn |
| Same transform mỗi epoch | Chưa xử lý epoch-aware seed | Quyết định ở training iteration, không sửa ngầm trong Phase 1 |
| Source-video leakage | Chưa xác minh | Tìm source ID/provenance; thêm group-aware split nếu có |
| Duplicate content khác tên | Chưa phát hiện | Cân nhắc content hash hoặc metadata index khi có dataset thật |
| Direct resize làm méo tỷ lệ | Chấp nhận tạm thời theo Section 0 | So sánh với letterbox/padding sau khi baseline ổn định |
| OpenCV reported frame count sai | Decoder sẽ fail rõ nếu decode thiếu frame | Inspection full decode và thống kê mismatch trên dữ liệu thật |
| Python/dependency compatibility | Chưa xác minh | Tạo clean environment với interpreter được dependency hỗ trợ và chạy install/test |
| Files đều untracked | Chưa có commit baseline | Review, sau đó người dùng tự quyết định commit |

---

## 11. Tiêu chí để coi Phase 1 hoàn tất đầy đủ

Phần code foundation đã có, nhưng Phase 1 chưa nên được coi là xác minh đầy đủ cho đến khi đạt các điểm sau:

1. Tạo clean Python environment.
2. Cài project runtime và test dependencies thành công.
3. `pytest -q` chạy collection đầy đủ.
4. Toàn bộ test cases pass, bao gồm DataLoader smoke test.
5. Có dataset path RWF-2000 thật.
6. Inspection report chạy thành công trên toàn dataset.
7. Xác nhận rõ directory nào là official train và official held-out test.
8. Xác nhận canonical label mapping đúng với layout thực tế.
9. Ghi nhận short/corrupt/decode-failure clips.
10. Tạo manifest bằng seed và validation fraction đã chốt.
11. Xác nhận manifest deterministic khi chạy lặp lại.
12. Xác nhận train/validation/test không overlap theo các metadata khả dụng.
13. Chạy DataLoader smoke test trên một số clip thật.
14. Ghi lại limitation source-level leakage nếu chưa có source IDs.

Chỉ sau các bước trên mới nên chuyển sang baseline ResNet18 + temporal average pooling.

---

## 12. Bước tiếp theo nhỏ nhất

### Bước 1 — Chuẩn bị environment và chạy tests

Tạo virtual environment bằng một Python interpreter tương thích với các dependency, sau đó:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest -q
```

Không nên tuyên bố test suite pass trước khi command cuối chạy thành công.

### Bước 2 — Cung cấp dataset path, không copy dataset vào Git

Ví dụ:

```text
D:\Datasets\RWF-2000
```

Dataset nên nằm ngoài repository hoặc trong thư mục `data/` đã được `.gitignore` bảo vệ.

### Bước 3 — Inspection trước manifest

```powershell
.\.venv\Scripts\python -m rwf2000 inspect `
  --root D:\Datasets\RWF-2000 `
  --splits train test `
  --output artifacts\rwf2000_inspection.json
```

Nếu layout thực là `train/val`, chưa được tự động coi `val` là tuning validation. Trước tiên phải xác nhận `val` có phải official held-out test của bản phân phối đó hay không.

### Bước 4 — Khóa manifest

Sau khi xác nhận split semantics:

```powershell
.\.venv\Scripts\python -m rwf2000 manifest `
  --root D:\Datasets\RWF-2000 `
  --output artifacts\rwf2000_manifest.csv `
  --train-split train `
  --test-split test `
  --val-fraction 0.2 `
  --seed 42
```

Seed `42` và validation fraction `0.2` hiện chỉ là default của code, chưa phải protocol được khóa bằng kết quả dataset thật hoặc quyết định cuối cùng.

### Bước 5 — Real-data smoke check

Sau khi manifest tồn tại:

- Load một vài records từ mỗi split.
- Decode đúng 16 ordered frames.
- Kiểm tra `[16, 3, 224, 224]` ở sample level.
- Kiểm tra `[B, 16, 3, 224, 224]` ở batch level.
- Kiểm tra label và metadata.
- Kiểm tra không có decode error.
- Kiểm tra output range sau normalization theo kỳ vọng.

### Bước 6 — Chỉ sau đó mới bắt đầu baseline

Baseline tiếp theo theo Section 0:

```text
16 frames
-> pretrained ResNet18 per frame
-> temporal average pooling
-> linear binary classifier
-> one violence logit
```

Temporal Transformer vẫn phải chờ baseline và evaluation foundation ổn định.

---

## 13. Danh sách file đã tạo trong iteration

### Config

- `pyproject.toml`

### Source

- `src/rwf2000/__init__.py`
- `src/rwf2000/__main__.py`
- `src/rwf2000/dataset.py`
- `src/rwf2000/decoder.py`
- `src/rwf2000/inspection.py`
- `src/rwf2000/labels.py`
- `src/rwf2000/manifest.py`
- `src/rwf2000/preprocessing.py`
- `src/rwf2000/sampling.py`

### Tests

- `tests/README.md`
- `tests/conftest.py`
- `tests/test_dataset.py`
- `tests/test_inspection.py`
- `tests/test_labels.py`
- `tests/test_manifest.py`
- `tests/test_sampling.py`

### Báo cáo trạng thái

- `PHASE1_RWF2000_IMPLEMENTATION_STATUS.md`

Không có existing source file nào bị refactor vì trước iteration repository chưa có source implementation.

---

## 14. Kết luận hiện trạng

Repository hiện có một foundation nhỏ, tách biệt và có contract tương đối rõ cho RWF-2000 data pipeline. Các rủi ro quan trọng như label inversion, temporal order, clip boundary, official-test contamination và filesystem-order nondeterminism đã được xử lý ở mức thiết kế/code/test source.

Tuy nhiên, trạng thái hiện tại vẫn là **implemented but not fully runtime-verified** vì:

- Test dependencies chưa được cài.
- Pytest chưa chạy.
- DataLoader smoke test chưa thực thi.
- OpenCV decode chưa chạy.
- RWF-2000 thật chưa được inspect.
- Source-video leakage chưa thể xác minh.

Không có cơ sở để báo cáo metric hoặc tuyên bố pipeline đã hoạt động trên RWF-2000 thật. Bước đúng tiếp theo là hoàn thiện environment, chạy full tests, sau đó inspect dataset thật và khóa manifest. Chưa nên bắt đầu model trước khi các bước này hoàn tất.

Nhánh anomaly Ped2/Avenue vẫn giữ nguyên trạng thái **bắt buộc trong sản phẩm cuối, chưa triển khai trong Phase 1**.
