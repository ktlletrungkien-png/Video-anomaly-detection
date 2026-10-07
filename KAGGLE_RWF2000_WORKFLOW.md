# Hướng dẫn RWF-2000 trên Kaggle — Phase 1 và Phase 2

## 1. Phạm vi của tài liệu

Tài liệu này hướng dẫn quy trình chuyển nhánh phát hiện bạo lực RWF-2000 từ repository local sang Kaggle: Phase 1 kiểm tra dữ liệu thật và Phase 2 chạy baseline `ResNet18 + temporal average pooling`.

Mục tiêu của quy trình:

```text
Repository local
-> GitHub
-> Kaggle Notebook
-> attach RWF-2000
-> kiểm tra layout và metadata
-> tạo manifest
-> Dataset/DataLoader smoke test
```

Phần Phase 1 chỉ gồm data-pipeline verification. Phần Phase 2 bắt đầu training sau khi code local và các sanity gate đã pass. Quy trình này:

- Không sửa core source trực tiếp trên Kaggle rồi bỏ quên repository local.
- Không chạy full training trước real-batch, one-batch overfit và pilot.
- Không implement Temporal Transformer.
- Không implement Ped2/Avenue.
- Không tải toàn bộ RWF-2000 về máy local.
- Không dùng synthetic fixtures để giả làm kết quả kiểm tra dữ liệu thật.

Nguồn định hướng có hiệu lực vẫn là `AGENTS.md` và Section 0 của `PROJECT_CONTEXT.md`.

## 2. Vai trò của local, GitHub và Kaggle

Ba môi trường có vai trò khác nhau:

```text
Local
-> viết và sửa source code
-> viết unit test
-> chạy kiểm tra nhanh
-> review diff
-> commit và push

GitHub
-> nguồn trung gian đồng bộ code đã review

Kaggle
-> attach dataset thật
-> chạy real-data verification
-> chạy workload cần tài nguyên cloud
-> lưu manifest, report, metric và checkpoint sau này
```

Repository local vẫn là nguồn code chính. Không nên sửa source chỉ trên Kaggle rồi để thay đổi tồn tại riêng trong notebook.

Nếu Kaggle phát hiện lỗi:

```text
Kaggle phát hiện lỗi
-> lưu error và đường dẫn clip gây lỗi
-> quay lại local
-> thêm test tái hiện lỗi nếu phù hợp
-> sửa source local
-> chạy pytest
-> review diff
-> commit và push
-> cập nhật repository trên Kaggle
-> chạy lại bước bị lỗi
```

## 3. Quy ước và nguyên tắc an toàn

- Dataset Kaggle được xem là input chỉ đọc dưới `/kaggle/input`.
- Artifact tạo ra phải được ghi dưới `/kaggle/working`.
- Không hard-code `/kaggle/input/...` vào core source trong `src/rwf2000`.
- Đường dẫn Kaggle chỉ được khai báo trong notebook hoặc truyền qua CLI.
- Không giả định split thật có tên `train`, `test` hoặc `val` trước khi nhìn thấy dataset.
- Official held-out partition phải được giữ bất biến.
- Validation chỉ được derive từ official training partition.
- Không dùng test labels để chọn threshold hoặc hyperparameter.
- Không commit dataset, checkpoint, token, credential, `.env` hoặc `kaggle.json`.
- Không sửa sampling/split policy để né lỗi dữ liệu trước khi review dữ liệu thật.
- Không khẳng định số clip, FPS, resolution hoặc chất lượng dataset trước khi đo.

## 4. Điều kiện hoàn thành Phase 1 trên Kaggle

Phase 1 real-data verification được xem là hoàn thành khi:

1. Repository cài được bằng `pip install -e ".[test]"`.
2. Toàn bộ unit tests pass trên Kaggle.
3. Dataset root, split names và class directories thật được xác nhận.
4. Inspection hoàn thành và lưu report.
5. Short clips và decode errors đã được review.
6. Manifest được tạo deterministic với seed cố định.
7. Official held-out partition vẫn thuộc derived `test`.
8. Một real sample có shape `[16, 3, 224, 224]`.
9. Một real batch có shape `[B, 16, 3, 224, 224]`.
10. Tensor hữu hạn, label hợp lệ và frame indices tăng theo thời gian.

## 5. Giai đoạn A — Chuẩn bị repository ở local

### Bước A1 — Kiểm tra trạng thái local

Chạy trong PowerShell tại repository local:

```powershell
git status --short
git branch --show-current
git log -1 --oneline
git remote -v
```

**Input**

- Repository local.
- Git metadata hiện tại.

**Output**

- Danh sách file đang thay đổi hoặc untracked.
- Branch hiện tại.
- Commit gần nhất.
- GitHub remote.

**Mục đích**

Biết chính xác code nào sẽ được đưa lên Kaggle. Không dùng `git add .` nếu chưa review các file untracked.

### Bước A2 — Chạy test local

Với virtual environment của project trên Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Input**

- Source code local.
- Synthetic unit-test fixtures.

**Output mong đợi**

- Tất cả test pass.
- Trạng thái local hiện tại: `68 passed`.
- Số test có thể tăng nếu repository được bổ sung sau này; dùng kết quả thực tế của commit đang chạy làm chuẩn.

**Mục đích**

Không đẩy một phiên bản đang lỗi lên Kaggle.

### Bước A3 — Commit và push code đã review

Người dùng tự chọn đúng file cần commit:

```powershell
git add <file-1> <file-2>
git status --short
git diff --cached
git commit -m "Mô tả thay đổi"
git push
```

**Input**

- Các thay đổi đã review.

**Output**

- Một commit mới trên GitHub.

**Mục đích**

Đảm bảo Kaggle clone đúng phiên bản source đã được kiểm tra.

Không commit dataset, checkpoint, credential hoặc file bí mật.

## 6. Giai đoạn B — Tạo Kaggle Notebook

### Bước B1 — Tạo notebook

Trên Kaggle:

1. Đăng nhập Kaggle.
2. Mở mục **Code**.
3. Chọn **New Notebook**.
4. Chọn Python Notebook.

Trong **Session options/Settings**:

- Accelerator: `None`.
- Internet: `On`.
- Language: `Python`.

**Input**

- Tài khoản Kaggle.

**Output**

- Một Kaggle Notebook trống.

**Mục đích**

Tạo môi trường Linux sạch. Phase 1 inspection chưa cần GPU; Internet cần thiết để clone GitHub và cài package nếu môi trường còn thiếu dependency.

### Bước B2 — Attach RWF-2000

Trong thanh bên của notebook:

1. Mở phần **Input**.
2. Chọn **Add Input**.
3. Tìm dataset RWF-2000 cần sử dụng.
4. Chọn **Add**.
5. Chấp nhận điều khoản nếu dataset yêu cầu.

**Input**

- Dataset RWF-2000 trên Kaggle.

**Output**

- Dataset xuất hiện bên dưới `/kaggle/input`.

**Mục đích**

Dùng dữ liệu thật trực tiếp trên Kaggle mà không tải toàn bộ dataset về PC.

## 7. Giai đoạn C — Kiểm tra môi trường và dataset layout

Mỗi đoạn Python dưới đây nên được đặt trong một code cell riêng và chạy lần lượt.

### Bước C1 — Kiểm tra runtime

```python
import platform
import sys

import cv2
import torch

print("Platform:", platform.platform())
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())
```

**Input**

- Kaggle Python environment.

**Output**

- Phiên bản Python, PyTorch và OpenCV.
- `CUDA available: False` là bình thường vì Accelerator đang là `None`.

**Mục đích**

Xác nhận runtime tối thiểu trước khi cài project.

### Bước C2 — Liệt kê các input đã attach

```python
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")

print("Attached inputs:")
for path in sorted(KAGGLE_INPUT.iterdir()):
    print(path)
```

**Input**

- `/kaggle/input`.

**Output**

- Một hoặc nhiều dataset directories, ví dụ minh họa:

```text
/kaggle/input/<dataset-slug>
```

**Mục đích**

Xác định dataset slug thật. Không đoán tên thư mục từ trước.

### Bước C3 — Xem cây thư mục ở độ sâu giới hạn

```python
MAX_DEPTH = 4

for path in sorted(KAGGLE_INPUT.rglob("*")):
    if path.is_dir():
        relative = path.relative_to(KAGGLE_INPUT)
        if len(relative.parts) <= MAX_DEPTH:
            print(path)
```

**Input**

- Dataset directories đã attach.

**Output**

- Cây thư mục tối đa bốn tầng.

**Mục đích**

Xác định:

- Dataset root thật.
- Official training partition.
- Official held-out partition.
- Class directories.
- Các lớp thư mục trung gian nếu có.

Không tiếp tục nếu chưa phân biệt được dataset root và split directories.

### Bước C4 — Đếm video và xem một số đường dẫn

```python
VIDEO_EXTENSIONS = {
    ".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"
}

video_paths = sorted(
    path
    for path in KAGGLE_INPUT.rglob("*")
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
)

print("Number of discovered videos:", len(video_paths))
print("\nFirst 20 video paths:")

for path in video_paths[:20]:
    print(path)
```

**Input**

- Files dưới `/kaggle/input`.

**Output**

- Tổng số video được discover trong tất cả input.
- Tối đa 20 video paths đầu tiên.

**Mục đích**

Kiểm tra dataset có video, extension được hỗ trợ, và cấu trúc split/class thể hiện như thế nào.

Con số ở đây chỉ mô tả input đã attach; chưa chứng minh đó là bản phát hành RWF-2000 chính thức.

### Bước C5 — Khai báo root và split names thật

Thay các placeholder bằng giá trị quan sát được ở các bước trước:

```python
RWF_ROOT = Path("/kaggle/input/<dataset-slug>/<actual-dataset-root>")
TRAIN_SPLIT = "<actual-official-training-directory>"
HELDOUT_SPLIT = "<actual-official-held-out-directory>"

OUTPUT_DIR = Path("/kaggle/working/rwf2000_phase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("RWF_ROOT:", RWF_ROOT)
print("TRAIN_SPLIT:", TRAIN_SPLIT)
print("HELDOUT_SPLIT:", HELDOUT_SPLIT)
print("OUTPUT_DIR:", OUTPUT_DIR)

assert RWF_ROOT.is_dir(), f"Dataset root does not exist: {RWF_ROOT}"
assert (RWF_ROOT / TRAIN_SPLIT).is_dir(), "Training split not found"
assert (RWF_ROOT / HELDOUT_SPLIT).is_dir(), "Held-out split not found"
assert TRAIN_SPLIT != HELDOUT_SPLIT
```

**Input**

- Dataset root và split names đã quan sát trực tiếp.

**Output mong đợi**

- Bốn giá trị được in ra.
- Không có `AssertionError`.

**Mục đích**

Chốt rõ official training và official held-out partitions mà không dựa vào assumption `train/test`.

### Bước C6 — Kiểm tra class directories

```python
for split_name in (TRAIN_SPLIT, HELDOUT_SPLIT):
    split_path = RWF_ROOT / split_name

    print(f"\nSplit: {split_name}")
    print(f"Path:  {split_path}")

    class_dirs = sorted(
        path.name
        for path in split_path.iterdir()
        if path.is_dir()
    )

    print("Class directories:", class_dirs)
```

**Input**

- Hai split directories thật.

**Output**

- Danh sách class directories trong từng split.

Pipeline hiện hỗ trợ exact aliases, không phân biệt chữ hoa/thường:

```text
Fight
Violence
NonFight
NonViolence
Non-violence
```

**Mục đích**

Xác nhận class mapping thật trước khi tạo manifest.

Nếu có class name khác, dừng lại. Không tự rename dataset và không sửa mapping trước khi review layout.

## 8. Giai đoạn D — Clone, cài và kiểm tra repository trên Kaggle

### Bước D1 — Clone repository

```python
import subprocess

REPO_URL = "https://github.com/ktlletrungkien-png/Video-anomaly-detection.git"
REPO_DIR = Path("/kaggle/working/Video-anomaly-detection")

if not (REPO_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        check=True,
    )
else:
    print("Repository already exists:", REPO_DIR)

subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True)
subprocess.run(["git", "status", "--short"], cwd=REPO_DIR, check=True)
```

**Input**

- GitHub repository public.

**Output**

- Repository tại `/kaggle/working/Video-anomaly-detection`.
- Commit hash hiện tại.
- `git status --short` không in gì nếu checkout sạch.

**Mục đích**

Đưa source đã review từ GitHub vào Kaggle.

### Bước D2 — Cài project

```python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
    cwd=REPO_DIR,
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    cwd=REPO_DIR,
    check=True,
)
```

**Input**

- `pyproject.toml`.
- Source package trong `src/rwf2000`.

**Output mong đợi**

- Package `rwf2000-pipeline` được cài editable.
- `pip check` báo không có broken requirements.

**Mục đích**

Kiểm tra repository có thể cài trong môi trường Linux sạch bằng packaging chính thức.

### Bước D3 — Chạy unit tests

```python
subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=REPO_DIR,
    check=True,
)
```

**Input**

- Source package vừa cài.
- Synthetic unit-test fixtures.

**Output mong đợi**

- Tất cả tests pass.
- Trạng thái local hiện tại có `68 passed`; số lượng có thể tăng sau này.

**Mục đích**

Phân biệt lỗi source/environment với lỗi chỉ xuất hiện khi dùng RWF-2000 thật.

Synthetic tests không phải bằng chứng thực nghiệm trên RWF-2000.

### Bước D4 — Kiểm tra CLI

```python
subprocess.run(
    [sys.executable, "-m", "rwf2000", "--help"],
    cwd=REPO_DIR,
    check=True,
)
```

**Input**

- Package đã cài.

**Output mong đợi**

- Hai subcommands: `inspect` và `manifest`.

**Mục đích**

Xác nhận `python -m rwf2000` hoạt động trên Kaggle/Linux.

## 9. Giai đoạn E — Real-data inspection

### Bước E1 — Chạy inspection

Lưu ý: inspection mở và decode video, vì vậy bước này sẽ lâu hơn unit tests.

```python
INSPECTION_JSON = OUTPUT_DIR / "inspection.json"

inspect_command = [
    sys.executable,
    "-m",
    "rwf2000",
    "inspect",
    "--root",
    str(RWF_ROOT),
    "--splits",
    TRAIN_SPLIT,
    HELDOUT_SPLIT,
    "--short-clip-length",
    "16",
    "--output",
    str(INSPECTION_JSON),
]

print("Inspection output:", INSPECTION_JSON)
subprocess.run(inspect_command, cwd=REPO_DIR, check=True)
```

**Input**

- Dataset root thật.
- Hai split names thật.
- Video thật.
- Short-clip threshold bằng 16 frame.

**Output**

- `/kaggle/working/rwf2000_phase1/inspection.json`.

**Mục đích**

Đo trực tiếp:

- Số clip theo split và class.
- Metadata frame count.
- Decoded frame count.
- FPS.
- Resolution.
- Duration.
- Clip dưới 16 frame.
- Video không mở được.
- Metadata/decode mismatch.

Đây là bước real-data verification đầu tiên.

### Bước E2 — Đọc inspection summary

```python
import json

with INSPECTION_JSON.open("r", encoding="utf-8") as file:
    inspection = json.load(file)

print("Dataset root:", inspection["dataset_root"])
print("Requested splits:", inspection["requested_splits"])
print("Short-clip threshold:", inspection["short_clip_length"])

for split_name, info in inspection["splits"].items():
    print(f"\n--- {split_name} ---")
    print("Number of clips:", info["num_clips"])
    print("Counts by label:", info["counts_by_label"])
    print("Short clips:", info["short_clips"])
    print("Decode failures/mismatches:", len(info["decode_failures"]))

    if info["decode_failures"]:
        print("First problematic paths:")
        for path in info["decode_failures"][:10]:
            print(" ", path)
```

**Input**

- `inspection.json`.

**Output**

- Tóm tắt count, short clips và decode problems theo split.

**Mục đích**

Quyết định dataset có đủ điều kiện để tạo manifest và Dataset sample hay chưa.

### Bước E3 — Xem phân bố FPS và resolution

```python
from collections import Counter

for split_name, info in inspection["splits"].items():
    opened_videos = [
        video
        for video in info["videos"]
        if video.get("opened")
    ]

    resolution_counts = Counter(
        (video.get("width"), video.get("height"))
        for video in opened_videos
    )

    fps_counts = Counter(
        round(video["fps"], 3)
        for video in opened_videos
        if video.get("fps") is not None
    )

    print(f"\n--- {split_name} ---")
    print("Most common resolutions:", resolution_counts.most_common(10))
    print("Most common FPS values:", fps_counts.most_common(10))
```

**Input**

- Per-video metadata trong inspection report.

**Output**

- Tối đa 10 resolution và FPS values phổ biến nhất theo split.

**Mục đích**

Phát hiện geometry, FPS hoặc metadata bất thường trước khi model được implement.

Không thay đổi preprocessing chỉ dựa trên phỏng đoán; trước tiên phải ghi nhận dữ liệu thật.

### Bước E4 — Gate trước manifest

```python
total_clips = sum(
    info["num_clips"]
    for info in inspection["splits"].values()
)

total_short_clips = sum(
    info["short_clips"]
    for info in inspection["splits"].values()
)

total_decode_problems = sum(
    len(info["decode_failures"])
    for info in inspection["splits"].values()
)

print("Total clips:", total_clips)
print("Total clips shorter than 16 decoded frames:", total_short_clips)
print("Total decode problems/mismatches:", total_decode_problems)
```

**Input**

- Inspection report.

**Output**

- Tổng số clip.
- Tổng short clips.
- Tổng decode problems/mismatches.

**Mục đích**

Tạo một điểm dừng rõ ràng trước khi manifest được sinh.

Chỉ tiếp tục ngay nếu:

- Tổng clip lớn hơn 0.
- Cả hai class được discover hợp lý.
- Split và class directories đúng.
- Không có decode problem nghiêm trọng.
- Không có clip dưới 16 decoded frames.

Nếu có short clip, pipeline hiện dùng strict policy và Dataset sẽ từ chối clip đó. Không tự padding, bỏ clip hoặc mượn frame từ clip khác trước khi review.

## 10. Giai đoạn F — Tạo và kiểm tra manifest

### Bước F1 — Tạo deterministic manifest

Chỉ chạy sau khi inspection đã được review.

```python
MANIFEST_CSV = OUTPUT_DIR / "rwf2000_manifest.csv"
MANIFEST_JSON = OUTPUT_DIR / "rwf2000_manifest.json"

manifest_command = [
    sys.executable,
    "-m",
    "rwf2000",
    "manifest",
    "--root",
    str(RWF_ROOT),
    "--train-split",
    TRAIN_SPLIT,
    "--test-split",
    HELDOUT_SPLIT,
    "--val-fraction",
    "0.2",
    "--seed",
    "42",
    "--output",
    str(MANIFEST_CSV),
]

subprocess.run(manifest_command, cwd=REPO_DIR, check=True)
```

**Input**

- Official training split.
- Official held-out split.
- Validation fraction `0.2`.
- Seed `42`.

**Output**

- `/kaggle/working/rwf2000_phase1/rwf2000_manifest.csv`.
- `/kaggle/working/rwf2000_phase1/rwf2000_manifest.json`.

**Mục đích**

- Giữ official held-out partition làm derived `test`.
- Derive validation chỉ từ official training partition.
- Split ở cấp clip và stratified theo class.
- Làm assignment deterministic.
- Lưu schema, seed, validation fraction và label mapping.

Tên option `--test-split` biểu thị official held-out partition, ngay cả khi tên thư mục thật không phải `test`.

### Bước F2 — Đọc manifest summary

```python
import csv

with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

with MANIFEST_JSON.open("r", encoding="utf-8") as file:
    manifest_metadata = json.load(file)

print("Number of manifest records:", len(rows))
print("Schema version:", manifest_metadata["schema_version"])
print("Seed:", manifest_metadata["seed"])
print("Validation fraction:", manifest_metadata["val_fraction"])
print("Label map:", manifest_metadata["label_map"])

counts = Counter(
    (row["derived_split"], row["label_name"])
    for row in rows
)

print("\nCounts by derived split and label:")
for key, count in sorted(counts.items()):
    print(key, count)
```

**Input**

- Manifest CSV và sidecar JSON.

**Output**

- Tổng records.
- Schema version.
- Seed.
- Validation fraction.
- Label mapping.
- Counts theo derived split và class.

**Mục đích**

Kiểm tra metadata tái lập và phân bố split trước khi decode sample.

### Bước F3 — Kiểm tra split protocol

```python
official_train_rows = [
    row for row in rows
    if row["source_split"] == TRAIN_SPLIT
]

official_heldout_rows = [
    row for row in rows
    if row["source_split"] == HELDOUT_SPLIT
]

assert official_train_rows
assert official_heldout_rows

assert all(
    row["derived_split"] in {"train", "val"}
    for row in official_train_rows
)

assert all(
    row["derived_split"] == "test"
    for row in official_heldout_rows
)

print("Protocol check passed:")
print("- Validation comes only from official training")
print("- Official held-out records remain in derived test")
```

**Input**

- Manifest rows.

**Output mong đợi**

- `Protocol check passed`.

**Mục đích**

Ngăn validation/test leakage do resplit sai official held-out partition.

## 11. Giai đoạn G — Real Dataset/DataLoader smoke test

### Bước G1 — Load một sample thật

```python
from rwf2000.dataset import RWF2000Dataset

train_dataset = RWF2000Dataset(
    dataset_root=RWF_ROOT,
    manifest=MANIFEST_CSV,
    split="train",
    training=False,
    seed=42,
    num_frames=16,
)

print("Number of training records:", len(train_dataset))

assert len(train_dataset) > 0

sample = train_dataset[0]

print("Frame tensor shape:", tuple(sample["frames"].shape))
print("Frame tensor dtype:", sample["frames"].dtype)
print("Label:", sample["label"].item())
print("Metadata:", sample["metadata"])
```

**Input**

- Dataset root thật.
- Manifest thật.
- Một derived training record.

**Output mong đợi**

```text
Frame tensor shape: (16, 3, 224, 224)
Frame tensor dtype: torch.float32
Label: 0.0 hoặc 1.0
Metadata: {...}
```

**Mục đích**

Xác minh end-to-end:

```text
manifest record
-> video path
-> frame count
-> 16 ordered indices
-> decode RGB
-> preprocessing
-> [16, 3, 224, 224]
```

### Bước G2 — Validate sample contract

```python
frames = sample["frames"]
label = int(sample["label"].item())
indices = sample["metadata"]["frame_indices"]

assert frames.shape == (16, 3, 224, 224)
assert frames.dtype == torch.float32
assert torch.isfinite(frames).all()

assert label in (0, 1)

assert len(indices) == 16
assert indices == sorted(indices)
assert all(
    left < right
    for left, right in zip(indices, indices[1:])
)

print("Real sample validation passed")
print("Label:", label)
print("Ordered frame indices:", indices)
```

**Input**

- Một sample thật đã decode.

**Output mong đợi**

- `Real sample validation passed`.
- Label và 16 ordered frame indices.

**Mục đích**

Xác nhận tensor shape, dtype, finite values, binary label và temporal order.

### Bước G3 — Tạo DataLoader batch

```python
from rwf2000.dataset import make_dataloader

batch_size = min(2, len(train_dataset))

train_loader = make_dataloader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    seed=42,
)

batch = next(iter(train_loader))

print("Batch frames shape:", tuple(batch["frames"].shape))
print("Batch frames dtype:", batch["frames"].dtype)
print("Batch labels:", batch["label"].tolist())
```

**Input**

- Real training Dataset.
- Tối đa hai records.

**Output mong đợi**

Nếu dataset có ít nhất hai training records:

```text
Batch frames shape: (2, 16, 3, 224, 224)
Batch frames dtype: torch.float32
Batch labels: [0.0/1.0, 0.0/1.0]
```

**Mục đích**

Xác nhận DataLoader contract `[B, 16, 3, 224, 224]`.

Dùng `num_workers=0` trong smoke test để decode error hiển thị rõ ràng. Chưa tối ưu throughput ở giai đoạn này.

### Bước G4 — Validate batch contract

```python
assert batch["frames"].ndim == 5
assert batch["frames"].shape[1:] == (16, 3, 224, 224)
assert batch["frames"].dtype == torch.float32
assert torch.isfinite(batch["frames"]).all()

assert set(batch["label"].tolist()).issubset({0.0, 1.0})

print("Real DataLoader smoke test passed")
```

**Input**

- Real batch.

**Output mong đợi**

- `Real DataLoader smoke test passed`.

**Mục đích**

Đánh dấu data pipeline đủ điều kiện chuyển sang bước chuẩn bị baseline. Đây chưa phải training và không phải metric mô hình.

## 12. Giai đoạn H — Lưu artifact và notebook version

### Bước H1 — Kiểm tra output files

```python
for path in sorted(OUTPUT_DIR.iterdir()):
    size_kb = path.stat().st_size / 1024
    print(f"{path.name}: {size_kb:.2f} KB")
```

**Input**

- `/kaggle/working/rwf2000_phase1`.

**Output mong đợi**

```text
inspection.json
rwf2000_manifest.csv
rwf2000_manifest.json
```

**Mục đích**

Xác nhận các artifact cần giữ nằm dưới `/kaggle/working`.

### Bước H2 — Lưu notebook reproducibly

Trên giao diện Kaggle:

1. Chọn **Save Version**.
2. Chọn **Save & Run All**.
3. Đặt tên, ví dụ `phase1-rwf2000-real-data-verification`.
4. Đảm bảo notebook outputs được lưu.
5. Chờ run hoàn thành từ đầu đến cuối.

**Input**

- Notebook cells.
- GitHub repository.
- Dataset đã attach.
- Các path variables đã cấu hình.

**Output**

- Một notebook version chạy trong session sạch.
- Artifacts từ `/kaggle/working` đi kèm version.

**Mục đích**

Chứng minh workflow không phụ thuộc state còn sót lại trong interactive session.

## 13. Các điều kiện phải dừng

Dừng trước manifest, model hoặc training nếu gặp bất kỳ trường hợp nào sau đây:

- Không xác định chắc chắn dataset root.
- Không rõ split nào là official training/held-out.
- Class directories không thuộc mapping hiện tại.
- Không tìm thấy video.
- Unit tests fail.
- Video không mở hoặc decode không hoàn chỉnh.
- Metadata frame count khác decoded frame count.
- Có clip dưới 16 decoded frames.
- Class count hoặc layout khác mô tả nguồn dataset.
- Manifest đưa official held-out record vào `train` hoặc `val`.
- Sample không có shape `[16, 3, 224, 224]`.
- Batch không có shape `[B, 16, 3, 224, 224]`.
- Label không thuộc `{0, 1}`.
- Tensor có `NaN` hoặc `Inf`.
- Frame indices không tăng nghiêm ngặt.

Khi dừng, cần lưu:

- Cell và command gây lỗi.
- Full traceback.
- Dataset path liên quan.
- Split/class name liên quan.
- Clip path liên quan nếu có.
- Inspection summary.

Sau đó quay lại local để quyết định sửa code hay điều chỉnh protocol dựa trên bằng chứng thật.

## 14. Cập nhật Kaggle sau khi sửa code local

Sau khi sửa, test, commit và push từ local, trên Kaggle chạy:

```python
subprocess.run(
    ["git", "pull", "--ff-only"],
    cwd=REPO_DIR,
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
    cwd=REPO_DIR,
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=REPO_DIR,
    check=True,
)
```

Nếu repository Kaggle đã bị sửa thủ công, nên dùng session sạch hoặc clone lại thay vì trộn thay đổi không được quản lý.

## 15. Checklist ngắn

### Local

- [ ] Đọc `AGENTS.md` và Section 0 của `PROJECT_CONTEXT.md`.
- [ ] `git status` đã được review.
- [ ] Unit tests local pass.
- [ ] Chỉ các file cần thiết được commit.
- [ ] Code mới nhất đã push GitHub.
- [ ] Không có dataset/checkpoint/credential trong commit.

### Kaggle setup

- [ ] Tạo Python Notebook.
- [ ] Accelerator để `None` cho Phase 1.
- [ ] Internet bật.
- [ ] RWF-2000 đã được attach.
- [ ] Dataset slug và directory tree đã được xem.
- [ ] `RWF_ROOT`, `TRAIN_SPLIT`, `HELDOUT_SPLIT` đã xác định từ dữ liệu thật.

### Repository verification

- [ ] Clone đúng repository.
- [ ] Ghi nhận commit hash.
- [ ] `pip install -e ".[test]"` thành công.
- [ ] `pip check` thành công.
- [ ] `pytest -q` pass.
- [ ] `python -m rwf2000 --help` hoạt động.

### Real-data verification

- [ ] Inspection hoàn thành.
- [ ] Counts theo split/class đã review.
- [ ] Short clips đã review.
- [ ] Decode errors/mismatches đã review.
- [ ] FPS/resolution đã review.
- [ ] Manifest được tạo với seed `42` và validation fraction `0.2`.
- [ ] Validation chỉ derive từ official training.
- [ ] Official held-out giữ nguyên trong derived `test`.
- [ ] Real sample validation pass.
- [ ] Real DataLoader smoke test pass.

### Output

- [ ] `inspection.json` tồn tại.
- [ ] `rwf2000_manifest.csv` tồn tại.
- [ ] `rwf2000_manifest.json` tồn tại.
- [ ] Notebook đã `Save & Run All` thành công.
- [ ] Chưa training và chưa tạo metric mô hình.

## 16. Trạng thái chuyển giao sau Phase 1

Nền tảng baseline local đã được implement và kiểm thử. Sau khi toàn bộ checklist Phase 1 pass:

1. Review diff local của Phase 2.
2. Commit và push đúng các file đã review.
3. Trên Kaggle, pull đúng commit mới.
4. Làm lần lượt các sanity gate và training steps trong mục 18.
5. Temporal Transformer chỉ bắt đầu sau khi baseline thật và evaluation artifacts ổn định.

Nhánh lightweight anomaly Ped2/Avenue vẫn là phần bắt buộc của sản phẩm cuối, nhưng không nằm trong Phase 1 RWF-2000 verification này.

## 17. Tài liệu Kaggle tham khảo

- Kaggle Notebooks: <https://www.kaggle.com/docs/notebooks>
- Kaggle Python Docker image: <https://github.com/Kaggle/docker-python>

Các chi tiết giao diện Kaggle có thể thay đổi theo thời gian. Các invariant quan trọng của workflow này là:

- Dataset được attach làm input.
- Source được lấy từ GitHub.
- Dataset root được truyền từ notebook/CLI.
- Outputs được ghi dưới `/kaggle/working`.
- Local vẫn là nguồn code chính.

## 18. Phase 2 — Chạy baseline trên Kaggle GPU

Chỉ bắt đầu phần này khi Phase 1 đã pass và code Phase 2 local đã được review, commit và push. Config pilot đã có trong repository; nó là điểm khởi đầu để đo hạ tầng, không phải final config.

### Bước P2.1 — Bật GPU và xác nhận runtime

Trong **Session options**, đổi Accelerator sang GPU. Sau khi session khởi động lại, chạy:

```python
import platform

import cv2
import matplotlib
import sklearn
import torch
import torchvision

print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

assert torch.cuda.is_available(), "Phase 2 training requires a Kaggle GPU session"
```

**Input**

- Kaggle GPU runtime.

**Output**

- Phiên bản dependency và tên GPU thật.
- `CUDA available: True`.

**Mục đích**

Ghi lại môi trường và xác nhận cặp PyTorch/Torchvision có sẵn hoạt động trước khi cài project. Không vội nâng cấp hai package lớn này trong notebook.

### Bước P2.2 — Cài đúng commit mà không thay cặp Torch/Torchvision của Kaggle

Sau khi clone đúng repository và checkout commit cần chạy:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[test]", "--no-deps"],
    cwd=REPO_DIR,
    check=True,
)

subprocess.run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=REPO_DIR,
    check=True,
)
```

Nếu cell import ở P2.1 cho thấy package nhỏ còn thiếu, chỉ cài package thiếu rồi restart session nếu Kaggle yêu cầu. Không thay Torch/Torchvision khi chưa có lý do và bằng chứng tương thích.

**Input**

- `REPO_DIR` trỏ tới đúng Git commit.
- Các dependency đã được kiểm tra ở P2.1.

**Output**

- Project được cài editable.
- Toàn bộ test pass trên Kaggle.

**Mục đích**

Đảm bảo code Kaggle giống code local và tránh resolver tải lại một cặp Torch/Torchvision lớn, không tương thích với base image.

### Bước P2.3 — Khóa đường dẫn Phase 1 và Phase 2

```python
from pathlib import Path

RWF_ROOT = Path("/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000")
MANIFEST = Path("/kaggle/input/<phase1-notebook-output>/rwf2000_manifest.csv")
PILOT_CONFIG = REPO_DIR / "configs" / "rwf2000_resnet18_avg_kaggle_pilot.json"
PILOT_OUTPUT = Path("/kaggle/working/rwf2000_phase2_pilot")

assert RWF_ROOT.is_dir()
assert MANIFEST.is_file()
assert PILOT_CONFIG.is_file()
assert not PILOT_OUTPUT.exists(), "Use a fresh output directory for a new pilot"
```

Nếu Phase 1 output chưa được attach làm Kaggle input, tạo lại manifest bằng đúng seed `42`, validation fraction `0.2`, source train `train` và held-out source `val`; không tự đổi protocol.

**Input**

- Dataset RWF-2000 thật.
- Manifest Phase 1 đã khóa.
- Pilot config từ repository.

**Output**

- Bốn path đã xác minh tồn tại hoặc chưa tồn tại đúng vai trò.

**Mục đích**

Ngăn notebook vô tình dùng manifest khác, config khác hoặc ghi đè một run cũ.

### Bước P2.4 — Real-batch forward

```python
import torch

from rwf2000.dataset import RWF2000Dataset, make_dataloader
from rwf2000.models import build_resnet18_temporal_average

train_dataset = RWF2000Dataset(
    RWF_ROOT,
    MANIFEST,
    split="train",
    training=False,
    seed=42,
    num_frames=16,
)
train_loader = make_dataloader(
    train_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    seed=42,
)

batch = next(iter(train_loader))
model = build_resnet18_temporal_average(
    weights="DEFAULT",
    freeze_backbone=True,
).cuda().eval()

with torch.no_grad():
    logits = model(batch["frames"].cuda())

print("frames:", tuple(batch["frames"].shape))
print("labels:", batch["label"].tolist())
print("logits:", logits.detach().cpu().tolist())
print("finite:", bool(torch.isfinite(logits).all()))

assert batch["frames"].shape == (2, 16, 3, 224, 224)
assert logits.shape == (2,)
assert torch.isfinite(logits).all()
```

**Input**

- Hai clip thật chỉ từ derived train.
- ImageNet-pretrained ResNet18.

**Output**

- Batch `[2,16,3,224,224]`.
- Hai raw logits hữu hạn.

**Mục đích**

Xác nhận decoder, preprocessing, GPU và model wiring hoạt động với dữ liệu thật trước khi tối ưu bất kỳ parameter nào.

### Bước P2.5 — One-batch overfit chỉ trên derived train

Dùng một batch cố định có cả hai lớp nếu có thể. Không lấy validation hoặc test cho sanity check này.

```python
import torch

fixed_batch = next(iter(make_dataloader(
    RWF2000Dataset(
        RWF_ROOT,
        MANIFEST,
        split="train",
        training=False,
        seed=42,
        num_frames=16,
    ),
    batch_size=8,
    shuffle=True,
    num_workers=2,
    seed=42,
)))

model = build_resnet18_temporal_average(
    weights="DEFAULT",
    freeze_backbone=True,
).cuda()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-2)
criterion = torch.nn.BCEWithLogitsLoss()
frames = fixed_batch["frames"].cuda()
labels = fixed_batch["label"].cuda()
losses = []

for step in range(30):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(frames), labels)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))

print("first loss:", losses[0])
print("last loss:", losses[-1])
assert losses[-1] < losses[0], "Stop: the fixed train batch did not overfit"
```

**Input**

- Một batch cố định từ derived train.
- Chỉ classifier head là trainable.

**Output**

- Chuỗi loss của 30 bước.
- Loss cuối thấp hơn loss đầu.

**Mục đích**

Phát hiện sớm lỗi label, loss, gradient, optimizer hoặc frozen-backbone wiring. Đây chỉ là sanity evidence, không phải metric mô hình.

### Bước P2.6 — Chạy pilot một epoch

```python
command = [
    sys.executable,
    "-m",
    "rwf2000",
    "train",
    "--root", str(RWF_ROOT),
    "--manifest", str(MANIFEST),
    "--config", str(PILOT_CONFIG),
    "--output", str(PILOT_OUTPUT),
    "--device", "cuda",
]

subprocess.run(command, cwd=REPO_DIR, check=True)
```

**Input**

- Derived train/validation trong manifest đã khóa.
- Config `kaggle_pilot_only_not_final` một epoch.

**Output**

- `run_config.json`, `environment.json`, `history.csv`.
- `best_checkpoint.pth`.
- Validation predictions/metrics và `selected_threshold.json`.
- Không load derived test.

**Mục đích**

Đo thời gian, GPU memory, DataLoader stability và kiểm tra artifact contract trước full training. Validation metric ở đây là pilot evidence, không phải final result.

### Bước P2.7 — Review pilot và khóa final config ở local

Kiểm tra:

```python
for name in (
    "run_config.json",
    "environment.json",
    "history.csv",
    "best_checkpoint.pth",
    "validation_predictions.csv",
    "validation_metrics.json",
    "selected_threshold.json",
):
    path = PILOT_OUTPUT / name
    print(name, path.exists(), path.stat().st_size if path.exists() else None)
```

**Input**

- Pilot artifacts.
- GPU memory/time quan sát từ Kaggle.

**Output**

- Quyết định có bằng chứng về batch size, workers, learning rate và số epoch cần thử tiếp.

**Mục đích**

Không biến giá trị phỏng đoán thành protocol. Mọi thay đổi config cần quay lại local, review, test, commit và push trước full run có chủ đích.

### Bước P2.8 — Full training sau khi config đã khóa

Dùng một output directory mới và file config final đã được commit. Chạy cùng lệnh `rwf2000 train` như P2.6, chỉ thay `--config` và `--output`.

**Input**

- Final config đã review.
- Derived train và derived validation; không có derived test trong training CLI.

**Output**

- Best checkpoint theo minimum validation loss.
- Threshold được chọn chỉ từ derived validation.
- Provenance hashes liên kết manifest, checkpoint và threshold.

**Mục đích**

Tạo baseline chính mà không dùng held-out labels để chọn model hoặc threshold.

### Bước P2.9 — Final held-out evaluation đúng một lần

Chỉ chạy sau khi đã khóa checkpoint và threshold của full run:

```python
FINAL_TRAIN_OUTPUT = Path("/kaggle/working/<full-run-output>")
FINAL_TEST_OUTPUT = Path("/kaggle/working/<full-run-test-output>")

command = [
    sys.executable,
    "-m",
    "rwf2000",
    "evaluate",
    "--root", str(RWF_ROOT),
    "--manifest", str(MANIFEST),
    "--checkpoint", str(FINAL_TRAIN_OUTPUT / "best_checkpoint.pth"),
    "--threshold", str(FINAL_TRAIN_OUTPUT / "selected_threshold.json"),
    "--output", str(FINAL_TEST_OUTPUT),
    "--device", "cuda",
]

subprocess.run(command, cwd=REPO_DIR, check=True)
```

**Input**

- Manifest, checkpoint và threshold cùng một run.
- Derived held-out test từ source `val`.

**Output**

- `test_predictions.csv` có probability, threshold và predicted label.
- `test_metrics.json` có Accuracy, Precision, Recall, F1, ROC-AUC và provenance hashes.
- `confusion_matrix.png`.

**Mục đích**

Đo same-domain held-out performance mà không retune sau khi nhìn test. CLI sẽ từ chối provenance mismatch và từ chối ghi đè final-test artifacts.

### Bước P2.10 — Lưu và báo cáo đúng mức bằng chứng

1. Save Version/Save & Run All để giữ `/kaggle/working` outputs.
2. Tải hoặc attach output thành Kaggle Dataset nếu cần giữ checkpoint.
3. Không commit checkpoint vào Git.
4. Báo validation và held-out test tách riêng.
5. Ghi rõ pilot metric không phải final metric.
6. Phân tích false positive/false negative từ `test_predictions.csv`.
7. Chỉ bắt đầu Temporal Transformer sau khi baseline artifacts và protocol đã được review.

**Input**

- Toàn bộ artifact của full run và final test.

**Output**

- Một run có thể truy vết tới Git commit, environment, config, manifest, checkpoint và threshold.

**Mục đích**

Giữ tính tái lập và tránh biến target, pilot hoặc external reference thành kết quả đo của project.
