# Kiến trúc multi-agent cho dự án Video Anomaly / Violence Detection

> Mục tiêu của tài liệu này là mô tả cách cấu hình **GPT-5.6 Sol xhigh** làm agent chính/orchestrator và **GPT-5.6 Luna xhigh** làm các subagent trong Codex, theo phạm vi **riêng của project**.
>
> Tài liệu này không thay thế `PROJECT_CONTEXT.md`. `PROJECT_CONTEXT.md` vẫn là nguồn bối cảnh nghiệp vụ/kỹ thuật chính của dự án; file này chỉ quy định cách các agent phối hợp để đọc, sửa, kiểm tra và đánh giá code.

---

## 1. Bối cảnh áp dụng

Hướng có hiệu lực hiện tại của dự án là:

- Phát hiện bạo lực trong video giám sát.
- Dataset chính: **RWF-2000**.
- Baseline: **ResNet18 + temporal average pooling**.
- Mô hình chính: **ResNet18 + Temporal Transformer**.
- Đầu ra chính: xác suất `Violence`, trạng thái cảnh báo, timeline xác suất và clip/snapshot của sự kiện.
- UBI-Fights có thể dùng làm external test nếu tiến độ cho phép.
- Ped2/Avenue vẫn tồn tại như nhánh bất thường tổng quát hoặc phần tham chiếu lịch sử, nhưng không được phép làm lệch hướng chính nếu xung đột với phần hiện hành trong `PROJECT_CONTEXT.md`.

Dự án còn ở giai đoạn đầu: ưu tiên data pipeline, baseline, sanity check và khả năng tái lập trước khi làm kiến trúc phức tạp.

Vì vậy multi-agent phải phục vụ bốn mục tiêu:

1. Giúp đọc và hiểu codebase nhanh hơn.
2. Tách việc kiểm tra data/model/evaluation thành các góc nhìn độc lập.
3. Giảm nguy cơ một agent vừa thiết kế, vừa implement, vừa tự xác nhận chính nó.
4. Không làm project phức tạp thêm nếu task nhỏ và một agent đã đủ.

---

## 2. Kiến trúc tổng thể

Kiến trúc đề xuất:

```text
                        USER
                          |
                          v
                GPT-5.6 Sol xhigh
                  MAIN ORCHESTRATOR
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
  project_explorer   ml_reviewer   experiment_reviewer
    Luna xhigh         Luna xhigh        Luna xhigh
     read-only          read-only         read-only
          \               |               /
           \              |              /
            +-------------+-------------+
                          |
                          v
                GPT-5.6 Sol xhigh
              phân tích / ra quyết định
                          |
                          v
               implementation_worker
                    Luna xhigh
                 workspace-write
                          |
                          v
                GPT-5.6 Sol xhigh
             review diff + validation
```

### Vai trò của Sol

Sol là agent chính và giữ quyền quyết định cuối cùng.

Sol chịu trách nhiệm:

- đọc yêu cầu của người dùng;
- xác định task có cần subagent hay không;
- chia task thành các phần độc lập;
- chọn subagent phù hợp;
- chờ kết quả;
- đối chiếu các kết quả mâu thuẫn;
- quyết định hướng sửa cuối cùng;
- giao một thay đổi hẹp cho worker;
- kiểm tra lại diff;
- chạy hoặc yêu cầu chạy validation cuối;
- không chấp nhận kết luận thực nghiệm nếu chưa có bằng chứng đo được.

### Vai trò của Luna

Luna được dùng cho các task hẹp, có phạm vi rõ và có thể chạy độc lập.

Luna không phải nơi đưa ra quyết định kiến trúc cuối cùng của project.

Nguyên tắc:

```text
Sol = planning + judgment + synthesis + final verification
Luna = exploration + review + bounded implementation
```

---

## 3. Vì sao dùng project-scoped config

Không đặt các custom agent này vào cấu hình global ngay từ đầu.

Cấu trúc nên nằm ngay trong repository:

```text
Video-anomaly-detection/
|
├── .codex/
│   ├── config.toml
│   └── agents/
│       ├── project_explorer.toml
│       ├── ml_reviewer.toml
│       ├── experiment_reviewer.toml
│       └── implementation_worker.toml
|
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── src/
├── configs/
├── notebooks/
├── demo/
├── tests/
└── ...
```

Lợi ích:

- chỉ project này nhìn thấy các custom agent;
- project khác không bị ảnh hưởng;
- có thể commit cấu hình cùng repository;
- các rule được version-control cùng code;
- dễ thay đổi khi project chuyển giai đoạn.

Việc một project có đăng ký subagent **không có nghĩa mọi chat/task trong project phải dùng subagent**.

Ví dụ:

```text
"Giải thích class này."
```

=> Sol tự xử lý.

```text
"Review toàn bộ data pipeline bằng các subagent độc lập."
```

=> Sol mới spawn Luna.

Nếu muốn chắc chắn không dùng subagent cho một task:

```text
Task này nhỏ. Không sử dụng subagent. Tự xử lý bằng main agent.
```

---

## 4. `.codex/config.toml`

Tạo:

```text
.codex/config.toml
```

Cấu hình đề xuất:

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"

[agents]
enabled = true
max_concurrent_threads_per_session = 4
default_subagent_model = "gpt-5.6-luna"
default_subagent_reasoning_effort = "xhigh"
```

Ý nghĩa:

- main agent mặc định: GPT-5.6 Sol;
- reasoning của main: xhigh;
- bật multi-agent;
- tối đa 4 subagent thread đồng thời;
- subagent mặc định dùng Luna;
- subagent mặc định cũng dùng xhigh.

Có thể dùng alias:

```toml
model = "gpt-5.6"
```

vì `gpt-5.6` hiện trỏ tới GPT-5.6 Sol.

### Tại sao chỉ để 4 thread

Project này không cần "đội quân" 10-20 agent.

Mục tiêu ban đầu:

- 2 agent: thường đủ;
- 3 agent: hợp lý cho review lớn;
- 4 agent: giới hạn an toàn để tránh lãng phí context/token và tăng coordination overhead.

Không tăng số lượng nếu chưa có một lý do rõ ràng để chia task độc lập.

---

## 5. `AGENTS.md`

`AGENTS.md` không nên chứa toàn bộ kiến thức của dự án.

Nó nên đóng vai trò **entry point / table of contents** để Codex biết phải đọc đâu và tuân thủ nguyên tắc gì.

Tạo file ở root:

```text
AGENTS.md
```

Nội dung gợi ý:

```md
# Project agent instructions

## Source of truth

Before making architectural or ML decisions, read `PROJECT_CONTEXT.md`.

The current effective project direction is section 0 of `PROJECT_CONTEXT.md`.
If historical sections conflict with section 0, section 0 wins.

## Current primary direction

The primary task is surveillance-video violence detection.

Main dataset:
- RWF-2000.

Current minimum model:
- ResNet18 + temporal average pooling.

Current main model:
- ResNet18 + Temporal Transformer.

Do not silently switch the project back to Ped2/Avenue anomaly detection.

## Experimental integrity

Never claim a metric, model improvement, dataset property observed locally,
or training result unless it was actually measured or supported by a source.

Do not tune using test labels.

Do not mix train/validation/test data.

Do not use test results to select model hyperparameters.

Distinguish clearly between:
- measured result;
- expected result;
- target;
- hypothesis.

## Repository safety

Do not commit:
- datasets;
- model checkpoints;
- credentials;
- access tokens;
- `.env`.

Prefer small, reviewable changes.

Do not perform unrelated refactors during a bounded task.

## Multi-agent policy

Use subagents only when the task benefits from independent or parallel work.

Prefer read-only subagents for:
- repository exploration;
- data-pipeline review;
- ML correctness review;
- evaluation review;
- test-gap analysis.

Only one write-capable implementation agent should own a bounded code change at a time.

The main Sol agent owns:
- task decomposition;
- final decisions;
- resolving conflicting findings;
- final diff review;
- final validation.

For small tasks, do not spawn subagents unless explicitly requested.
```

---

## 6. Custom subagent 1: `project_explorer`

Tạo:

```text
.codex/agents/project_explorer.toml
```

Nội dung:

```toml
name = "project_explorer"
description = "Read-only repository explorer for locating implementations, tracing execution paths, and collecting evidence before changes are proposed."

model = "gpt-5.6-luna"
model_reasoning_effort = "xhigh"
sandbox_mode = "read-only"

developer_instructions = """
Act as a read-only codebase explorer.

Before drawing conclusions:
- inspect the relevant repository files;
- trace real call paths;
- identify configs, tests, scripts, and entry points;
- cite concrete file paths and symbols;
- distinguish current code from plans described only in documentation.

For this project:
- read PROJECT_CONTEXT.md when project intent matters;
- section 0 is the current direction when history conflicts;
- do not assume planned modules already exist.

Do not edit files.
Do not implement fixes.
Do not expand scope.

Return to the parent:
1. relevant files/symbols;
2. current execution/data flow;
3. concrete evidence;
4. unresolved questions or risks.
"""
```

### Khi dùng

Ví dụ:

```text
Hãy dùng project_explorer để tìm luồng từ Dataset -> DataLoader ->
model input và chỉ ra tensor shape tại từng bước.
```

Hoặc:

```text
Cho project_explorer xác định code nào hiện thực sự đã tồn tại,
code nào mới chỉ được mô tả trong PROJECT_CONTEXT.md.
```

---

## 7. Custom subagent 2: `ml_reviewer`

Tạo:

```text
.codex/agents/ml_reviewer.toml
```

Nội dung:

```toml
name = "ml_reviewer"
description = "Read-only ML reviewer focused on data pipelines, tensor shapes, preprocessing, model correctness, leakage, and train/inference consistency."

model = "gpt-5.6-luna"
model_reasoning_effort = "xhigh"
sandbox_mode = "read-only"

developer_instructions = """
Review ML code conservatively.

Focus on:
- train/validation/test separation;
- dataset split correctness;
- label mapping;
- temporal frame ordering;
- clip boundaries;
- tensor dimensions;
- preprocessing consistency;
- normalization;
- augmentation validity;
- loss/output compatibility;
- sigmoid/logit mistakes;
- frozen vs trainable parameters;
- train/eval mode;
- device/dtype mismatches;
- reproducibility;
- potential leakage.

For violence detection:
- verify Violence/Non-violence labels are not reversed;
- verify frame sampling preserves temporal order;
- verify evaluation uses the intended split;
- verify thresholds are not chosen using test labels.

For the historical anomaly-detection branch:
- do not treat it as the active main direction unless the parent explicitly asks.

Do not edit files.

Return findings ordered by severity with:
- file/symbol;
- problem;
- why it matters;
- evidence;
- suggested direction.
"""
```

### Khi dùng

Ví dụ:

```text
Cho ml_reviewer review Dataset/DataLoader mới,
đặc biệt kiểm tra leakage, label mapping, clip boundary và tensor shape.
```

---

## 8. Custom subagent 3: `experiment_reviewer`

Tạo:

```text
.codex/agents/experiment_reviewer.toml
```

Nội dung:

```toml
name = "experiment_reviewer"
description = "Read-only experiment reviewer focused on metrics, split protocol, reproducibility, configuration, checkpoints, and whether conclusions are supported by measured evidence."

model = "gpt-5.6-luna"
model_reasoning_effort = "xhigh"
sandbox_mode = "read-only"

developer_instructions = """
Audit experiments and evaluation logic.

Focus on:
- train/validation/test protocol;
- threshold selection;
- checkpoint selection;
- random seeds;
- config capture;
- reproducibility;
- metric implementation;
- class imbalance handling;
- confusion matrix correctness;
- Accuracy / Balanced Accuracy / Precision / Recall / F1 / ROC-AUC;
- false alarms per hour when implemented;
- inference latency and FPS when claimed;
- same-domain vs external-test conclusions.

Reject unsupported claims.

Never convert a target or estimate from PROJECT_CONTEXT.md into a measured result.

Check that:
- test labels are not used for tuning;
- external UBI-Fights results are not used to tune the model;
- RWF-2000 same-domain and external evaluation are clearly separated.

Do not edit files.

Return:
1. protocol issues;
2. metric issues;
3. reproducibility gaps;
4. unsupported conclusions;
5. recommended corrections.
"""
```

### Khi dùng

Ví dụ:

```text
Cho experiment_reviewer audit evaluate.py và configs hiện tại.
Tập trung vào threshold, checkpoint selection, metric và data leakage.
```

---

## 9. Custom subagent 4: `implementation_worker`

Tạo:

```text
.codex/agents/implementation_worker.toml
```

Nội dung:

```toml
name = "implementation_worker"
description = "Write-capable implementation agent for small, already-understood and well-bounded changes."

model = "gpt-5.6-luna"
model_reasoning_effort = "xhigh"
sandbox_mode = "workspace-write"

developer_instructions = """
Implement only the bounded task assigned by the parent orchestrator.

Before editing:
- inspect the relevant code;
- understand the requested behavior;
- preserve current public behavior outside the task scope.

Rules:
- make the smallest defensible change;
- do not redesign the architecture unless explicitly asked;
- do not make unrelated refactors;
- do not change dataset protocol silently;
- do not invent experimental results;
- do not commit datasets, checkpoints, credentials, or secrets;
- add or update focused tests when appropriate.

Validation:
- run targeted tests or smoke checks appropriate for the change;
- do not launch expensive full-dataset training unless explicitly requested;
- report commands run and their outcomes;
- report all files changed.

If the task is ambiguous, stop and report the ambiguity to the parent instead of guessing.
"""
```

### Khi dùng

Không gọi worker ngay từ đầu cho task phức tạp.

Luồng ưu tiên:

```text
explore/review
    ->
Sol xác định vấn đề
    ->
worker sửa một scope hẹp
    ->
Sol review diff
```

---

## 10. Quy tắc write ownership

Đây là rule quan trọng nhất của kiến trúc.

### Nên

```text
project_explorer        read-only
ml_reviewer             read-only
experiment_reviewer     read-only

             |
             v

Sol quyết định một plan duy nhất

             |
             v

implementation_worker   write
```

### Không nên

```text
worker A sửa dataset.py
worker B sửa dataset.py
worker C sửa train.py liên quan cùng interface

-> merge conflict
-> assumption khác nhau
-> khó biết thay đổi nào gây lỗi
```

Nếu cần nhiều worker song song, chỉ làm khi file ownership tách hoàn toàn và Sol xác định dependency rõ ràng.

Ở giai đoạn hiện tại của project, mặc định:

> **Một writer tại một thời điểm.**

---

## 11. Khi nào nên dùng subagent

### Nên dùng

Task có nhiều hướng kiểm tra độc lập, ví dụ:

```text
Review toàn bộ RWF-2000 data pipeline.
```

Có thể chia:

```text
project_explorer
-> map implementation và data flow

ml_reviewer
-> kiểm tra preprocessing, labels, leakage, tensor shapes

experiment_reviewer
-> kiểm tra split, config, reproducibility
```

Sau đó Sol tổng hợp.

### Nên dùng

```text
Baseline train ra accuracy gần 50%. Tìm nguyên nhân.
```

Chia:

- explorer: trace pipeline;
- ML reviewer: labels / loss / logits / preprocessing;
- experiment reviewer: split / metric / checkpoint / evaluation.

### Nên dùng

```text
Review một thay đổi lớn trước khi merge.
```

Các reviewer độc lập có thể phát hiện lỗi khác nhau.

### Không cần dùng

```text
Giải thích BCEWithLogitsLoss.
```

### Không cần dùng

```text
Đổi tên một biến.
```

### Không cần dùng

```text
Sửa một import.
```

### Không nên dùng nhiều writer

```text
Implement toàn bộ Transformer bằng 4 worker cùng lúc.
```

Ở project này, cách đó dễ tạo architecture không thống nhất hơn là tiết kiệm thời gian.

---

## 12. Prompt mẫu: review data pipeline

```text
Review data pipeline hiện tại của RWF-2000 bằng subagents.

Dùng song song:
- project_explorer: map toàn bộ flow từ video -> sampling -> Dataset ->
  DataLoader -> tensor đưa vào model;
- ml_reviewer: kiểm tra label mapping, clip boundary, temporal order,
  preprocessing, augmentation, tensor shape và leakage;
- experiment_reviewer: kiểm tra split, config, reproducibility và việc
  validation/test có bị dùng sai hay không.

Tất cả subagent chỉ đọc, không sửa code.

Chờ cả ba hoàn thành.

Sau đó chính bạn, với vai trò main agent:
1. hợp nhất các finding;
2. loại bỏ duplicate;
3. giải quyết finding mâu thuẫn bằng cách kiểm tra code thực tế;
4. xếp hạng severity;
5. đề xuất plan sửa tối thiểu.

Chưa implement cho đến khi plan đã rõ.
```

---

## 13. Prompt mẫu: debug model không học

```text
Baseline hiện đang học kém.

Dùng subagents để điều tra độc lập, không sửa code:

- project_explorer:
  trace train loop, model forward, dataset output và evaluation flow.

- ml_reviewer:
  kiểm tra label mapping, logits, BCEWithLogitsLoss, sigmoid usage,
  optimizer, requires_grad, train/eval mode, preprocessing,
  tensor shape và khả năng overfit một batch.

- experiment_reviewer:
  kiểm tra split, metric, checkpoint selection, random seed và
  validation/test protocol.

Chờ tất cả hoàn thành.

Sau đó hãy:
1. lập bảng các hypothesis;
2. gắn evidence cho từng hypothesis;
3. ưu tiên các nguyên nhân có thể kiểm chứng nhanh nhất;
4. chỉ sau khi đủ bằng chứng mới giao implementation_worker sửa
   một nguyên nhân tại một thời điểm;
5. chạy sanity check nhỏ trước, không chạy full training ngay.
```

---

## 14. Prompt mẫu: implement sau khi đã hiểu vấn đề

```text
Chúng ta đã xác định lỗi là label mapping Violence/Non-violence bị đảo.

Giao implementation_worker:
- chỉ sửa mapping;
- thêm test regression nhỏ;
- không refactor Dataset;
- không đổi split hoặc augmentation;
- chạy test liên quan.

Sau khi worker hoàn thành:
- tự đọc diff;
- xác minh test;
- kiểm tra xem thay đổi có ảnh hưởng config hoặc checkpoint cũ không;
- báo lại chính xác file nào thay đổi.
```

---

## 15. Prompt mẫu: không dùng subagent

```text
Task này nhỏ.
Không sử dụng subagent.
Chỉ dùng main Sol xhigh để giải thích đoạn code này.
Không sửa file.
```

Hoặc:

```text
Không delegate.
Hãy tự sửa typo này và chạy test tối thiểu liên quan.
```

---

## 16. Workflow khuyến nghị theo từng giai đoạn dự án

### Giai đoạn A - Khởi tạo data pipeline

Ưu tiên:

```text
Sol
├── project_explorer
├── ml_reviewer
└── implementation_worker
```

Các lỗi data pipeline có thể làm toàn bộ training sai, nên review kỹ hơn model architecture.

### Giai đoạn B - Baseline ResNet18 + temporal average

Ưu tiên:

```text
Sol
├── ml_reviewer
├── experiment_reviewer
└── implementation_worker
```

Bắt buộc có sanity check:

- model forward chạy;
- shape đúng;
- loss giảm trên batch nhỏ;
- model có thể overfit một batch nhỏ;
- metric pipeline hoạt động.

### Giai đoạn C - Temporal Transformer

Không để worker tự thay đổi nhiều phần cùng lúc.

Ưu tiên:

```text
Sol thiết kế
    ->
project_explorer xác nhận interface hiện tại
    ->
ml_reviewer kiểm tra shape / temporal semantics
    ->
implementation_worker implement scope nhỏ
    ->
Sol review và test
```

### Giai đoạn D - Evaluation

Ưu tiên:

```text
Sol
├── experiment_reviewer
└── ml_reviewer
```

Kiểm tra đặc biệt:

- threshold chỉ lấy từ validation;
- test không dùng để tune;
- same-domain và external test tách biệt;
- Accuracy không bị dùng sai trên dataset mất cân bằng;
- không biến expected range thành actual result.

### Giai đoạn E - Demo

Có thể dùng:

```text
project_explorer
-> map inference pipeline

implementation_worker
-> implement bounded UI/inference change
```

Sol vẫn review:

- model input;
- smoothing;
- threshold;
- persistence rule;
- saved clip;
- latency/FPS claim.

---

## 17. Context giữa các agent

Subagent có context riêng.

Không nên giả định rằng mọi Luna đều "biết tất cả những gì Sol đang nghĩ".

Parent/orchestrator phải giao task rõ:

```text
Mục tiêu
+
phạm vi
+
file/khu vực cần xem nếu đã biết
+
điều không được làm
+
output cần trả về
```

Ví dụ tốt:

```text
Review src/datasets/rwf2000.py và code liên quan.

Mục tiêu:
xác minh 16 frame được sampling đúng temporal order.

Không sửa code.

Kiểm tra:
- clip boundary;
- short clip behavior;
- frame ordering;
- tensor shape;
- train/test consistency.

Trả về findings có file/symbol/evidence.
```

Không nên giao:

```text
Xem thử code có vấn đề gì không.
```

---

## 18. Context chung của project

`PROJECT_CONTEXT.md` là source of truth cho intent của project.

Tuy nhiên không nên copy toàn bộ file vào mọi prompt.

Dùng `AGENTS.md` để chỉ đường:

```text
Read PROJECT_CONTEXT.md when project intent matters.
Section 0 is authoritative when history conflicts.
```

Agent chỉ cần đọc phần cần thiết.

Đây là cách giữ context gọn hơn so với nhồi toàn bộ lịch sử dự án vào mỗi subagent.

---

## 19. Nguyên tắc tránh hallucination trong project ML

Agent phải phân biệt bốn loại thông tin:

| Loại | Ví dụ |
|---|---|
| Đã đo | `validation F1 = 0.84` từ output thực tế |
| Đã xác minh trong code | Dataset trả tensor `[B, T, C, H, W]` |
| Mục tiêu | mong muốn accuracy 82-88% |
| Giả thuyết | Transformer có thể cải thiện temporal modeling |

Không được biến:

```text
"Mục tiêu 82-88%"
```

thành:

```text
"Model đạt 85%"
```

Nếu chưa chạy training/test, kết luận phải ghi rõ là chưa đo.

---

## 20. Training nặng và môi trường cloud

Local Codex phù hợp cho:

- code;
- unit test;
- smoke test;
- shape test;
- overfit batch nhỏ nếu phần cứng cho phép;
- config;
- lint;
- evaluation trên mẫu nhỏ.

Không tự động chạy full training nhiều giờ chỉ vì một subagent muốn kiểm chứng giả thuyết.

Với training nặng trên Colab/Kaggle:

```text
local implementation
    ->
local smoke test
    ->
commit/push
    ->
Colab/Kaggle training
    ->
save result/checkpoint externally
    ->
đưa metric/log nhẹ về project
    ->
Sol + experiment_reviewer phân tích
```

Dataset và checkpoint lớn không đưa vào Git.

---

## 21. Strategy mặc định cho project này

Cấu hình vận hành khuyến nghị:

```text
MAIN
GPT-5.6 Sol
reasoning = xhigh

SUBAGENTS
GPT-5.6 Luna
reasoning = xhigh

MAX CONCURRENT
4

DEFAULT POLICY
- task nhỏ -> Sol tự làm
- task lớn, đọc nhiều -> parallel read-only Luna
- implementation -> một Luna writer
- final decision -> Sol
- final diff review -> Sol
```

Tóm tắt:

```text
                 Sol xhigh
            understand + plan
                  |
           cần parallel?
             /          \
           không          có
            |             |
        Sol tự làm     Luna read-only
                          |
                    collect findings
                          |
                       Sol xhigh
                    final decision
                          |
                   cần sửa code?
                     /       \
                   không       có
                    |          |
                  done    Luna worker
                               |
                          Sol review
                               |
                         validation
                               |
                             done
```

---

## 22. Trình tự triển khai thực tế

### Bước 1

Tạo:

```text
.codex/config.toml
```

### Bước 2

Tạo:

```text
.codex/agents/
```

### Bước 3

Tạo bốn agent:

```text
project_explorer.toml
ml_reviewer.toml
experiment_reviewer.toml
implementation_worker.toml
```

### Bước 4

Tạo `AGENTS.md` ở root.

### Bước 5

Mở project bằng Codex mới để cấu hình project-scoped được load.

### Bước 6

Xác nhận main model đang là:

```text
GPT-5.6 Sol
Extra High
```

### Bước 7

Test bằng một task read-only:

```text
Dùng project_explorer và ml_reviewer song song để đọc project hiện tại.
Không sửa code.

project_explorer:
- xác định repository hiện đã implement đến đâu.

ml_reviewer:
- xác định các phần ML chưa tồn tại hoặc mới chỉ là kế hoạch.

Chờ cả hai rồi tổng hợp.
```

### Bước 8

Kiểm tra subagent threads trong UI để chắc chắn:

- agent đúng role;
- dùng Luna;
- reasoning đúng;
- read-only agent không sửa file.

### Bước 9

Chỉ sau khi read-only flow ổn mới thử `implementation_worker`.

---

## 23. Tiêu chí đánh giá kiến trúc multi-agent có đáng dùng hay không

Sau vài task, không đánh giá bằng cảm giác "có nhiều agent nên mạnh hơn".

Theo dõi:

- thời gian hoàn thành;
- số finding thực sự hữu ích;
- finding trùng nhau;
- số lần Sol phải sửa lại output của Luna;
- số conflict khi edit;
- số token/credit tiêu thụ;
- số lỗi được phát hiện trước khi chạy training dài;
- mức độ rõ ràng của final diff.

Nếu task nhỏ mà:

```text
multi-agent cost > single-agent benefit
```

thì dùng Sol một mình.

Multi-agent là công cụ để phân rã công việc, không phải chế độ bắt buộc.

---

## 24. Kiến trúc cuối cùng đề xuất

```text
Video-anomaly-detection/
|
├── AGENTS.md
├── PROJECT_CONTEXT.md
|
├── .codex/
│   ├── config.toml
│   └── agents/
│       ├── project_explorer.toml
│       ├── ml_reviewer.toml
│       ├── experiment_reviewer.toml
│       └── implementation_worker.toml
|
├── src/
├── configs/
├── notebooks/
├── demo/
├── tests/
└── README.md
```

Mô hình trách nhiệm:

```text
GPT-5.6 Sol xhigh
        |
        |--- project_explorer      -> hiểu code thực tế
        |
        |--- ml_reviewer           -> kiểm tra ML/data/model
        |
        |--- experiment_reviewer   -> kiểm tra experiment/metric/leakage
        |
        `--- implementation_worker -> implement thay đổi đã được giới hạn
        |
        v
GPT-5.6 Sol xhigh
final integration + validation
```

Đây là kiến trúc khởi đầu. Không thêm agent mới cho đến khi xuất hiện một nhóm công việc lặp lại đủ rõ để biện minh cho một role riêng.

---

## 25. Tài liệu chính thức liên quan

- Codex subagents và custom agents:
  `https://learn.chatgpt.com/docs/agent-configuration/subagents`
- `AGENTS.md`:
  `https://learn.chatgpt.com/docs/agent-configuration/agents-md`
- GPT-5.6 Sol:
  `https://developers.openai.com/api/docs/models/gpt-5.6-sol`
- GPT-5.6 Luna:
  `https://developers.openai.com/api/docs/models/gpt-5.6-luna`

> Lưu ý: format custom-agent TOML là một capability đang tiếp tục được phát triển. Nếu Codex báo một key không hợp lệ sau khi client được cập nhật, ưu tiên schema/tài liệu chính thức của phiên bản Codex đang dùng.
