# PROJECT CONTEXT

## Hệ thống giám sát phát hiện bạo lực và bất thường trong video CCTV

> Tài liệu tổng hợp bối cảnh, mục tiêu, quyết định kỹ thuật, đánh giá kế hoạch và trạng thái thực tế của dự án.  
> Cập nhật gần nhất: 07/10/2026.
> Nội dung và sản phẩm presentation được chủ động loại khỏi tài liệu này.

---

## 0. HƯỚNG PHÁT TRIỂN HIỆN TẠI

> **Quyết định ngày 10/09/2026, cập nhật ngày 12/09/2026:** dự án theo hướng **hệ thống lai**. Nhánh chính là phát hiện bạo lực, đặc biệt là gây gổ và đánh nhau; nhánh bất thường tổng quát cũng phải có trong phiên bản nộp nhưng bắt đầu bằng phương pháp nhẹ. Mô hình bất thường lớn thứ hai chỉ được huấn luyện khi phần cốt lõi đã hoàn thành và còn thời gian. Phần kế hoạch Ped2/Avenue bên dưới được giữ lại để bảo toàn lịch sử thảo luận; khi có xung đột, phần 0 này là thông tin có hiệu lực.

### 0.1. Tên đề tài đề xuất

> Xây dựng hệ thống giám sát video lai phát hiện hành vi bạo lực và bất thường bằng CNN kết hợp Temporal Transformer.

Tên tiếng Anh có thể dùng:

> Hybrid Violence and Anomaly Detection in Surveillance Videos using CNN and Temporal Transformer.

### 0.2. Mục tiêu

Xây dựng hệ thống nhận video từ camera hoặc video file, phân tích các đoạn ngắn và đưa ra hai loại tín hiệu: xác suất xảy ra bạo lực và mức độ khác biệt so với hành vi bình thường của camera. Phiên bản đầu tập trung vào bạo lực thể chất có chuyển động rõ, ví dụ:

- Đánh nhau bằng tay hoặc chân.
- Đấm, đá, xô đẩy mạnh.
- Nhiều người tham gia ẩu đả.
- Một người tấn công người khác.

Hệ thống không cố gắng suy luận ý định, lời nói đe dọa hoặc xung đột không có biểu hiện hình ảnh rõ ràng.

Đầu ra dự kiến:

- Xác suất `Violence` cho mỗi cửa sổ video.
- Anomaly score cho mỗi cửa sổ video của nhánh bất thường tổng quát.
- Trạng thái `Normal`, `Warning: unknown anomaly` hoặc `Violence detected`.
- Thời gian bắt đầu và kết thúc sự kiện.
- Ảnh đại diện và clip được lưu khi cảnh báo.
- Đồ thị xác suất theo thời gian.

### 0.3. Thay đổi bản chất bài toán

Kế hoạch cũ là normal-only anomaly detection:

```text
Chỉ học video bình thường -> anomaly score
```

Hướng mới là supervised binary video classification hoặc temporal violence detection:

```text
Video có nhãn Violence / Non-violence
-> model học đặc trưng của hai lớp
-> xác suất bạo lực
```

Ưu điểm của hướng mới:

- Mục tiêu rõ ràng hơn.
- Có thể dùng Accuracy, Precision, Recall và F1 một cách trực tiếp.
- Cảnh báo có nguyên nhân rõ hơn “bất thường chưa xác định”.
- Dễ xây demo có tính ứng dụng.
- Phù hợp hơn nếu mục tiêu cuối là phát hiện gây gổ, đánh nhau.

Đánh đổi của nhánh bạo lực:

- Chỉ phát hiện tốt các mẫu bạo lực tương tự dữ liệu huấn luyện.
- Một mình nhánh này không có khả năng kết luận các bất thường chưa biết.
- Có thể nhầm chơi thể thao, ôm, nhảy hoặc chuyển động mạnh thành bạo lực.
- Cross-dataset vẫn là bài toán khó do khác camera, chất lượng video và bối cảnh.

### 0.4. Dataset chính

**Cập nhật yêu cầu ngày 10/09/2026:** dự án được phép sử dụng nhiều dataset, hai dataset chỉ là mức tối thiểu. Vì vậy, hệ thống lai có thể dùng dataset riêng cho nhánh bạo lực và nhánh bất thường tổng quát. Không cần ép hai nhiệm vụ khác nhau vào cùng một cặp dataset.

**RWF-2000** được chọn làm dataset chính.

- 2.000 clip lấy từ cảnh quay giám sát trong bối cảnh thực tế.
- 1.000 clip bạo lực và 1.000 clip không bạo lực.
- Phù hợp cho binary classification.
- Dataset cân bằng nên Accuracy dễ diễn giải hơn so với dataset anomaly mất cân bằng.
- Công trình gốc công bố Flow Gated Network đạt 87,25% accuracy trên test set. Đây là kết quả tham khảo, không phải kết quả được bảo đảm cho dự án.

Nguồn:

- https://arxiv.org/abs/1911.05913

### 0.5. Danh sách dataset cho hệ thống lai

Ba dataset cốt lõi được đề xuất:

| Dataset | Vai trò | Cách sử dụng |
|---|---|---|
| RWF-2000 | Nhánh phát hiện bạo lực | Train, validation và test theo split cố định với nhãn Violence/Non-violence |
| UCSD Ped2 | Nhánh bất thường tổng quát | Học normality từ Ped2 Train và đánh giá trên Ped2 Test |
| CUHK Avenue | Xác nhận nhánh bất thường | Học normality từ Avenue Train và đánh giá trên Avenue Test |

Dataset thứ tư tùy tiến độ là **UBI-Fights**, dùng để external test nhánh bạo lực.

- 1.000 video, tổng thời lượng khoảng 80 giờ.
- 216 video chứa sự kiện đánh nhau và 784 video bình thường.
- Có frame-level annotation.
- Bao gồm indoor, outdoor, camera cố định, camera quay và video màu hoặc grayscale.
- Gần bối cảnh giám sát hơn Hockey Fight.

Nguồn:

- https://socia-lab.di.ubi.pt/EventDetection/

Chiến lược cho nhánh bạo lực:

| Vai trò | Dataset | Cách dùng |
|---|---|---|
| Training và validation | RWF-2000 Train | Học Violence/Non-violence và chọn threshold |
| Kiểm tra chính | RWF-2000 Test | Đánh giá same-domain để xác nhận model hoạt động |
| Kiểm tra tổng quát hóa | UBI-Fights | External test nếu còn thời gian, không tune bằng test labels |

Chiến lược cho nhánh bất thường:

| Camera profile | Dữ liệu học normality | Dữ liệu kiểm tra |
|---|---|---|
| Ped2 | Ped2 Train | Ped2 Test |
| Avenue | Avenue Train | Avenue Test |

Không gộp Ped2 Train và Avenue Train thành một normal memory ở phiên bản đầu. Mỗi camera hoặc dataset có một normal profile riêng để giảm ảnh hưởng của domain shift.

Nếu yêu cầu của giảng viên bắt buộc “một dataset train, một dataset test” theo nghĩa tuyệt đối:

```text
RWF-2000 -> training và validation
UBI-Fights -> testing duy nhất
```

Cách này là cross-dataset evaluation và rủi ro kết quả thấp hơn rõ rệt. Cần xác nhận lại yêu cầu với giảng viên trước khi khóa protocol. Phương án same-domain RWF-2000 Test vẫn nên được giữ làm sanity check nội bộ, ngay cả khi không phải kết quả chính nộp cho bài toán cross-dataset.

UCF-Crime và XD-Violence chưa được chọn cho phiên bản đầu:

- UCF-Crime có 1.900 video dài, 128 giờ và 13 loại bất thường với nhãn yếu theo video.
- XD-Violence có 4.754 video, 217 giờ, nhiều nguồn video và cả âm thanh.
- Hai dataset này phù hợp temporal localization hoặc weakly supervised learning nhưng nặng hơn nhiều cho người mới và thời hạn 8 tuần.

### 0.6. Kiến trúc hệ thống lai

Hệ thống gồm hai nhánh có độ ưu tiên khác nhau. Nhánh bạo lực là phần mô hình học sâu chính và phải được huấn luyện đầy đủ. Nhánh bất thường nhẹ cũng là phần bắt buộc để trả về cảnh báo `unknown anomaly`; nó không cần huấn luyện một mạng lớn thứ hai trong phiên bản cơ sở.

```text
Video window
├─ Nhánh bạo lực: ResNet18 + Temporal Transformer -> P(Violence)
└─ Nhánh bất thường: đặc trưng normal profile -> anomaly score

P(Violence) cao                  -> Violence detected
P(Violence) không cao, score cao -> Warning: unknown anomaly
Hai tín hiệu đều thấp             -> Normal
```

#### 0.6.1. Nhánh bạo lực — bắt buộc

Baseline tối thiểu:

```text
16 frame
-> ResNet18 pretrained cho từng frame
-> temporal average pooling
-> linear classifier
-> P(Violence)
```

Mô hình chính:

```text
16 frame
-> ResNet18 pretrained lấy feature từng frame
-> positional encoding
-> Temporal Transformer 2 lớp, 4 attention heads
-> temporal pooling
-> binary classification head
-> P(Violence)
```

Cấu hình ban đầu:

- Input: 16 frame lấy đều từ một clip hoặc sliding window.
- Resolution khởi đầu: 224 × 224, sau đó kiểm tra phương án giữ tỷ lệ bằng padding.
- Backbone: ResNet18 pretrained.
- Giai đoạn đầu freeze phần lớn backbone.
- Loss: Binary Cross Entropy hoặc BCEWithLogitsLoss.
- Optimizer: AdamW.
- Transformer: 2 layers, 4 heads.
- Output: một logit, sau sigmoid thành xác suất bạo lực.
- Augmentation: horizontal flip, brightness/contrast nhẹ, random crop có kiểm soát.
- Không dùng augmentation đảo thứ tự frame hoặc tạo chuyển động phi thực tế.

Ablation bắt buộc:

```text
ResNet18 + temporal average
so với
ResNet18 + Temporal Transformer
```

So sánh này trả lời trực tiếp Transformer có giúp học chuyển động theo thời gian hay không.

YOLO không phải thành phần bắt buộc của phiên bản đầu. Có thể thêm YOLO hoặc person detector ở giai đoạn mở rộng để:

- Phát hiện vùng có người.
- Crop vùng tương tác để giảm ảnh hưởng nền.
- Hiển thị bounding box hỗ trợ giải thích.

YOLO một mình không thể kết luận hai người đang đánh nhau.

#### 0.6.2. Nhánh bất thường tổng quát nhẹ — bắt buộc

Phiên bản cơ sở dùng ResNet18 đã pretrained để lấy embedding cho từng video window, sau đó tạo **normal profile riêng** cho Ped2 và Avenue từ các video train bình thường. Anomaly score có thể là khoảng cách từ embedding mới đến normal memory/centroid (cosine distance, Mahalanobis distance, k-NN hoặc One-Class SVM). Cách này đủ để hiện thực hóa nhánh cảnh báo bất thường mà không phải train thêm một model lớn.

```text
Video window
-> ResNet18 pretrained lấy embedding
-> so sánh với normal profile của camera/dataset tương ứng
-> anomaly score
```

Không gộp normal profile của Ped2 và Avenue. Khi demo bằng camera thật, cần thu 30–60 phút video bình thường của chính camera đó để tạo hoặc hiệu chỉnh normal profile.

#### 0.6.3. Mô hình bất thường lớn thứ hai — chỉ thực hiện khi còn thời gian

Sau khi hoàn thành, đánh giá và tích hợp hai nhánh cơ sở, có thể huấn luyện **một mô hình temporal anomaly detection lớn thứ hai** trên từng dataset Ped2/Avenue theo split riêng. Chọn **một** trong hai phương án sau, không làm đồng thời:

| Phương án | Luồng gợi ý | Lưu ý |
|---|---|---|
| CNN + Temporal Transformer | Chuỗi feature CNN -> Transformer -> dự đoán feature/frame tiếp theo -> prediction error | Phù hợp để so sánh với Transformer của nhánh bạo lực; tốn thời gian thiết kế/evaluate hơn |
| ConvLSTM-Autoencoder | Chuỗi frame -> encoder -> ConvLSTM -> decoder -> reconstruction error | Dễ giải thích theo lỗi tái tạo; có thể nhạy với ánh sáng, nền và ảnh mờ |

Mô hình lớn này thay thế hoặc đối chiếu với normal-profile nhẹ ở nhánh bất thường, **không thay thế nhánh phát hiện bạo lực**. Chỉ bắt đầu khi các tiêu chí ở mục 0.11, trừ mục mở rộng này, đã đạt và vẫn còn buffer cho đánh giá, demo và báo cáo.

### 0.7. Pipeline ứng dụng

```text
Video file / webcam / RTSP
-> đọc frame
-> sliding window 16 frame
-> nhánh bạo lực và nhánh anomaly profile
-> violence probability + anomaly score
-> temporal smoothing
-> rule hợp nhất trạng thái
-> alert, log and saved clip
```

Quy tắc cảnh báo khởi đầu có thể là:

- Model tính xác suất cho một window mỗi 0,5 đến 1 giây.
- Nếu `P(Violence)` vượt ngưỡng, ưu tiên trạng thái `Violence detected`.
- Nếu xác suất bạo lực không vượt ngưỡng nhưng anomaly score vượt ngưỡng, tạo `Warning: unknown anomaly`.
- Làm mượt xác suất bằng exponential moving average.
- Chọn threshold bằng validation, không mặc định dùng 0,5.
- Chỉ cảnh báo khi nhiều window liên tiếp vượt threshold.
- Dùng cooldown để tránh tạo nhiều cảnh báo cho cùng một vụ việc.

Các giá trị threshold, số window liên tiếp và cooldown chưa được chốt. Chúng phải được chọn bằng validation và kiểm tra trên video demo.

### 0.8. Metric mới

Metric bắt buộc cho nhánh bạo lực:

- Accuracy trên dataset cân bằng.
- Balanced Accuracy khi dataset mất cân bằng như UBI-Fights.
- Precision cho lớp Violence.
- Recall cho lớp Violence.
- F1-score cho lớp Violence.
- ROC-AUC.
- Confusion matrix.

Metric bắt buộc cho nhánh bất thường nhẹ:

- Frame-level AUROC và Average Precision (AP) trên Ped2 Test và Avenue Test theo profile riêng.
- Minh họa ít nhất một trường hợp cảnh báo đúng và một false alarm.

Metric hệ thống:

- Số cảnh báo giả trên mỗi giờ video.
- Event-level recall.
- Độ trễ cảnh báo.
- FPS và inference latency.

Trong bài toán an toàn, Recall cho Violence quan trọng vì bỏ sót vụ việc có hậu quả lớn. Tuy nhiên, không được tăng Recall bằng cách tạo quá nhiều cảnh báo giả. F1 và false alarms/hour cần được báo cáo cùng Recall.

### 0.9. Dự đoán kết quả để lập kế hoạch

Đây là ước lượng, không phải cam kết:

| Kịch bản | Khoảng dự kiến |
|---|---:|
| ResNet18 + temporal average trên RWF-2000 same-domain | 75 đến 85% accuracy |
| ResNet18 + Temporal Transformer trên RWF-2000 same-domain | 82 đến 90% accuracy |
| Triển khai tốt và tuning hợp lý | khoảng 85 đến 90% accuracy |
| Train RWF-2000, test UBI-Fights không adaptation | khoảng 60 đến 78% balanced accuracy hoặc ROC-AUC |

Công trình RWF-2000 gốc báo cáo 87,25% accuracy bằng Flow Gated Network. Mục tiêu hợp lý cho dự án là:

```text
Tối thiểu:   >= 75% accuracy trên RWF-2000 test
Mục tiêu:    82 đến 88%
Tốt:         >= 88%
Không cam kết: > 90%
```

Nếu same-domain RWF-2000 chỉ đạt khoảng 50%, phải xem đó là lỗi hoặc mô hình chưa học được. Các nguyên nhân cần kiểm tra gồm label sai, sampling sai, class mapping bị đảo, model không overfit được batch nhỏ hoặc preprocessing train/test không nhất quán.

### 0.10. Lộ trình 8 tuần sau khi đổi hướng

| Tuần | Trọng tâm | Đầu ra |
|---|---|---|
| 1 | Khảo sát violence detection | Phạm vi, dataset, kiến trúc và metric |
| 2 | RWF-2000 data pipeline | Thống kê, clip sampler, augmentation và DataLoader |
| 3 | Baseline | ResNet18 + temporal average, checkpoint và confusion matrix |
| 4 | Temporal Transformer | Model chính chạy được và overfit sanity check |
| 5 | Training và tuning | Validation, threshold và checkpoint tốt nhất |
| 6 | Đánh giá bạo lực và nhánh anomaly nhẹ | RWF-2000 test, ablation; normal profile Ped2/Avenue và metric anomaly theo split riêng |
| 7 | Demo lai | Video input, xác suất, anomaly score, hợp nhất trạng thái, alert, log và saved clip |
| 8 | Hoàn thiện và phần mở rộng có điều kiện | Báo cáo, kiểm tra tái lập, bảo vệ; chỉ thử Transformer/ConvLSTM anomaly nếu còn buffer |

### 0.11. Tiêu chí thành công mới

Dự án được xem là đạt mục tiêu tối thiểu khi:

1. RWF-2000 data pipeline hoạt động đúng.
2. Baseline train và inference được.
3. CNN-Transformer tạo xác suất Violence cho video window.
4. Có so sánh công bằng với baseline.
5. Có Accuracy, Precision, Recall, F1, ROC-AUC và confusion matrix.
6. Same-domain RWF-2000 test đạt mức có ý nghĩa, mục tiêu thực tế từ 82 đến 88% accuracy.
7. Demo nhận video và tạo cảnh báo theo thời gian.
8. Có phân tích false positive và false negative.
9. Có báo cáo FPS, latency và false alarms/hour nếu đủ thời gian.
10. Nhánh normal-profile nhẹ chạy và được đánh giá trên Ped2 và Avenue theo split riêng của từng dataset.
11. External test trên UBI-Fights được xem là mục tiêu nâng cao, không phải điều kiện tối thiểu nếu việc truy cập hoặc preprocessing quá nặng.
12. CNN-Transformer hoặc ConvLSTM lớn cho nhánh bất thường là mục tiêu mở rộng có điều kiện, không được làm chậm việc hoàn thành hệ thống lai cơ sở.

### 0.12. Việc cần làm ngay

**Cập nhật ngày 07/10/2026:** Phase 2 RWF-2000 baseline đã hoàn thành trên Kaggle; Gate A-G đều pass. Model là pretrained frozen ResNet18 + temporal average pooling, train 5 epochs trên 1.280 derived-train clips, chọn checkpoint/threshold bằng 320 derived-validation clips và chỉ sau khi khóa mới evaluate 400 derived-test clips. Held-out result đo được: Accuracy `0.74`, Precision `0.712389`, Recall `0.805`, F1 `0.755869`, ROC-AUC `0.814525`; confusion matrix `[[135,65],[39,161]]`. Threshold validation là `0.4555857181549072`. Đây là kết quả project đã đo, không phải target; accuracy thấp hơn planning target tối thiểu 75% một điểm phần trăm. Không được retune bằng test set hoặc chạy lại test để chọn kết quả đẹp hơn.

Việc cần làm tiếp theo:

1. Commit artifact nhẹ và báo cáo Phase 2; checkpoint được backup riêng ngoài Git, nhận dạng bằng SHA-256 `c1e16c3aa93d560978c225bc213f2a42ab51a16255caed801602d40f0c7c6078`.
2. Implement Phase 3 `ResNet18 + Temporal Transformer` ở local, giữ nguyên manifest, 16 frames, resolution, preprocessing, loss và evaluation protocol để tạo ablation công bằng.
3. Transformer chỉ dùng derived train/validation để chọn model và threshold; không dùng baseline test errors hoặc Transformer test labels để tune.
4. Chạy local tests, real-batch forward, one-batch overfit và pilot trước full Transformer training.
5. Evaluate Transformer held-out test đúng một lần sau khi config/checkpoint/threshold đã khóa.
6. So sánh baseline và Transformer bằng Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix và failure cases.
7. Sau khi nhánh violence chính ổn định, xây normal profile nhẹ riêng cho Ped2 rồi Avenue.
8. Chỉ cân nhắc mô hình anomaly lớn khi hai nhánh cơ sở, evaluation và demo đã ổn định.

Tài liệu chi tiết:

- `PHASE2_RWF2000_BASELINE_PLAN.md`.
- `results/rwf-2000/phase2_resnet18_temporal_avg/README.md`.
- `PHASE3_RWF2000_TEMPORAL_TRANSFORMER_PLAN.md`.

---

## 1. Tóm tắt điều hành của kế hoạch trước, chỉ giữ để tham chiếu lịch sử

Dự án thuộc học phần **Thiết kế hệ thống thông minh**, thời gian dự kiến khoảng **8 tuần**. Chủ đề ban đầu là “xây dựng hệ thống trinh sát phát hiện hành vi bất thường”. Sau khi phân tích, phạm vi phù hợp và khả thi hơn được xác định là:

> Xây dựng hệ thống hỗ trợ phát hiện đối tượng và chuyển động bất thường trong video từ camera cố định tại khu vực dành cho người đi bộ, sử dụng CNN kết hợp Temporal Transformer.

Hệ thống học từ video chứa hoạt động bình thường. Khi nhận video mới, hệ thống tính **anomaly score** cho từng frame, làm mượt điểm theo thời gian và phát cảnh báo khi điểm vượt ngưỡng trong nhiều frame liên tiếp.

Định hướng hiện tại:

- **Baseline bắt buộc:** Conv-Autoencoder.
- **Mô hình chính:** CNN, ưu tiên ResNet18 pretrained, kết hợp Temporal Transformer.
- **Mô hình mở rộng:** ConvLSTM-Autoencoder, chỉ thực hiện nếu còn thời gian.
- **Dataset phát triển chính:** UCSD Ped2.
- **Dataset thứ hai:** CUHK Avenue.
- **Đánh giá chính:** theo split chính thức của từng dataset.
- **Đánh giá nâng cao:** train trên Ped2 và test trên Avenue để khảo sát cross-domain.
- **Sản phẩm cuối:** code, checkpoint, kết quả đánh giá, demo video/camera và báo cáo.

Đánh giá chung về kế hoạch ban đầu:

- Phù hợp cho một đồ án nghiên cứu hoặc proof of concept.
- Khả thi trong 8 tuần nếu giảm số mô hình và số ablation.
- Chưa đủ để gọi là hệ thống ứng dụng nếu chỉ có notebook và AUROC.
- Cần bổ sung pipeline đầu vào camera, xử lý ngưỡng, giảm báo giả, lưu sự kiện và đo tốc độ.
- Điểm rủi ro lớn nhất là dùng Ped2 để train rồi test trực tiếp trên Avenue như thí nghiệm duy nhất.

---

## 2. Bối cảnh người thực hiện

- Người thực hiện biết lập trình.
- Chưa có kiến thức nền tảng về AI và Deep Learning.
- Cần học theo hướng thực hành, chỉ học những kiến thức phục vụ trực tiếp cho dự án.
- Nền tảng huấn luyện dự kiến là Google Colab Pro hoặc Kaggle GPU.
- Máy cá nhân dùng để viết code, kiểm tra dữ liệu nhỏ, quản lý Git và viết báo cáo.
- GitHub dùng để đồng bộ code giữa máy cá nhân và môi trường cloud.
- Dataset và checkpoint có dung lượng lớn phải lưu trên Google Drive hoặc Kaggle Datasets, không commit vào Git.

Hệ quả đối với phạm vi:

- Không nên tự xây ba mô hình phức tạp cùng lúc.
- Không nên đặt mục tiêu vượt state of the art.
- Cần hoàn thành data pipeline và baseline trước Transformer.
- Ưu tiên một hệ thống chạy hoàn chỉnh hơn một kiến trúc quá phức tạp nhưng chưa tích hợp được.

---

## 3. Vấn đề cần giải quyết

### 3.1. Bối cảnh thực tế

Camera giám sát hoạt động liên tục và tạo ra lượng video lớn. Người vận hành khó duy trì sự tập trung khi phải xem nhiều luồng camera. Hệ thống AI có thể hỗ trợ bằng cách sàng lọc video, đánh dấu thời điểm khác thường và lưu lại bằng chứng để con người kiểm tra.

Hệ thống chỉ đóng vai trò **hỗ trợ cảnh báo**. Con người vẫn xác minh sự kiện và đưa ra quyết định. Dự án không xử lý nhận dạng khuôn mặt hoặc danh tính.

### 3.2. Phát biểu bài toán

Cho trước tập video từ camera cố định chỉ chứa hoạt động bình thường, xây dựng mô hình học đặc trưng không gian và thời gian của hoạt động đó. Khi tiếp nhận một video mới, mô hình phải:

1. Tính anomaly score cho từng frame hoặc từng đoạn ngắn.
2. Xác định thời điểm có dấu hiệu bất thường.
3. Nếu có thể, tạo anomaly map biểu diễn vùng đóng góp vào điểm bất thường.
4. Phát cảnh báo sau khi áp dụng ngưỡng và quy tắc thời gian.
5. Lưu thời gian, điểm số, ảnh đại diện và đoạn video liên quan.

### 3.3. Định nghĩa “bất thường”

Bất thường phụ thuộc vào bối cảnh. Ví dụ, xe đạp bình thường trên đường giao thông nhưng bất thường trong lối đi chỉ dành cho người đi bộ.

Với Ped2 và Avenue, phạm vi hợp lý gồm:

- Xe đạp, skateboard hoặc xe nhỏ xuất hiện ở khu vực dành cho người đi bộ.
- Người chạy hoặc di chuyển khác với mẫu chuyển động thường gặp.
- Đi sai hướng hoặc xuất hiện vật thể khác thường.
- Một số hành vi lạ trong Avenue nếu ground truth có hỗ trợ.

Không nên tuyên bố hệ thống có thể phát hiện đầy đủ:

- Trộm cắp.
- Ý định phạm tội.
- Mọi hình thức bạo lực.
- Mọi hành vi đáng ngờ trong mọi bối cảnh.

Nếu mục tiêu thực sự là đánh nhau, cướp, bạo lực hoặc tai nạn, Ped2 không phù hợp. Khi đó cần chuyển sang bài toán weakly supervised video anomaly detection và các dataset như UCF-Crime hoặc XD-Violence. Hướng này nặng hơn đáng kể và không phải lựa chọn mặc định cho kế hoạch 8 tuần hiện tại.

---

## 4. Phạm vi hệ thống

### 4.1. Trong phạm vi

- Camera cố định.
- Video file là đầu vào đầu tiên.
- Webcam hoặc RTSP là đầu vào mở rộng.
- Xử lý chuỗi frame theo sliding window.
- Tính anomaly score theo frame.
- Làm mượt score theo thời gian.
- Cảnh báo khi nhiều frame liên tiếp vượt ngưỡng.
- Hiển thị video kèm score và trạng thái cảnh báo.
- Lưu nhật ký sự kiện.
- Đánh giá bằng metric học thuật và metric vận hành.

### 4.2. Ngoài phạm vi phiên bản đầu

- Nhận dạng khuôn mặt hoặc danh tính.
- Hiểu ý định của con người.
- Phân loại chính xác tên của mọi hành vi bất thường.
- Theo dõi đồng thời số lượng lớn camera.
- Bảo đảm không có cảnh báo giả.
- Triển khai production ở quy mô tổ chức.

### 4.3. Yêu cầu chức năng sơ bộ

| Mã | Yêu cầu |
|---|---|
| FR-01 | Đọc video từ file |
| FR-02 | Có khả năng mở rộng sang webcam hoặc RTSP |
| FR-03 | Resize, normalize và tạo chuỗi frame |
| FR-04 | Tính anomaly score cho từng frame |
| FR-05 | Làm mượt score theo thời gian |
| FR-06 | Phát cảnh báo theo ngưỡng và số frame liên tiếp |
| FR-07 | Hiển thị score và trạng thái trên video |
| FR-08 | Lưu thời gian, score, ảnh và đoạn video của sự kiện |
| FR-09 | Chạy đánh giá tự động trên dataset có ground truth |

### 4.4. Yêu cầu phi chức năng sơ bộ

| Mã | Yêu cầu |
|---|---|
| NFR-01 | Kết quả thực nghiệm có thể tái lập bằng config và random seed |
| NFR-02 | Báo cáo FPS, latency và tài nguyên sử dụng |
| NFR-03 | Không đưa dataset, checkpoint hoặc bí mật truy cập vào Git |
| NFR-04 | Không sử dụng nhận dạng khuôn mặt |
| NFR-05 | Lưu nguồn và giấy phép của dataset |
| NFR-06 | Tách rõ dữ liệu train, validation và test để tránh leakage |

---

## 5. Thiết lập học máy

### 5.1. Cách học

Phương án chính là **normal-only learning**, thường được các tài liệu gọi là unsupervised video anomaly detection hoặc one-class video anomaly detection.

Trong quá trình training:

- Model chỉ xem video bình thường.
- Model học hình ảnh và chuyển động thường gặp.
- Nhãn bất thường không tham gia tối ưu model.

Trong quá trình testing:

- Video có thể chứa cả bình thường và bất thường.
- Frame khác nhiều với quy luật đã học nhận score cao.
- Ground truth chỉ dùng để đo kết quả.

Thuật ngữ “unsupervised” cần được giải thích cẩn thận. Training không dùng nhãn bất thường, nhưng dataset đã biết trước tập train chứa hoạt động bình thường và nhãn test vẫn được dùng để đánh giá.

### 5.2. Các khái niệm AI tối thiểu cần nắm

- **Tensor:** cấu trúc dữ liệu nhiều chiều được PyTorch sử dụng.
- **Dataset:** lớp mô tả cách đọc một mẫu dữ liệu.
- **DataLoader:** tạo batch, trộn dữ liệu và cung cấp mẫu cho quá trình train.
- **Batch:** một nhóm mẫu được xử lý cùng lúc.
- **Epoch:** một lần model học hết tập train.
- **Feature:** biểu diễn do model rút ra từ ảnh hoặc chuỗi frame.
- **Forward:** quá trình đưa dữ liệu qua model để tạo output.
- **Loss:** số đo model đang sai bao nhiêu.
- **Backward:** tính gradient để biết cách cập nhật tham số.
- **Optimizer:** cập nhật tham số dựa trên gradient.
- **Inference:** sử dụng model đã train để dự đoán trên dữ liệu mới.
- **Overfitting:** model nhớ tập train nhưng hoạt động kém trên dữ liệu khác.
- **Anomaly score:** mức độ bất thường, score cao biểu thị mẫu ít giống dữ liệu bình thường.

---

## 6. Dataset

## 6.1. UCSD Ped2

Thông tin chính:

- Camera cố định đặt ở vị trí cao, quan sát lối đi bộ.
- Độ phân giải khoảng 360 × 240 pixel.
- 16 clip training.
- 12 clip testing.
- Training chỉ chứa hoạt động bình thường.
- Testing chứa cả frame bình thường và bất thường.
- Bất thường thường gặp gồm xe đạp, skateboard, xe nhỏ và chuyển động không đúng bối cảnh.
- Có ground truth theo frame và vùng ảnh.

Vai trò dự kiến:

- Dataset phát triển chính.
- Dùng để hoàn thiện data pipeline và baseline.
- Dùng để huấn luyện và đánh giá mô hình chính theo split chính thức.
- Phù hợp với Colab/Kaggle và người mới vì dung lượng nhỏ.

Hạn chế:

- Chỉ đại diện cho một camera và một bối cảnh.
- Số lượng video nhỏ.
- Các bất thường tương đối đơn giản.
- Kết quả cao trên Ped2 không chứng minh khả năng triển khai trên camera khác.

Nguồn:

- https://www.svcl.ucsd.edu/projects/anomaly/
- https://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf

## 6.2. CUHK Avenue

Thông tin chính:

- Video màu, quay tại lối đi trong khuôn viên CUHK.
- 16 video training.
- 21 video testing.
- Tổng cộng 30.652 frame, gồm 15.328 frame training và 15.324 frame testing.
- Training chứa các tình huống bình thường.
- Testing chứa cả bình thường và bất thường.
- Có ground truth theo thời gian và bounding box cho vùng bất thường.
- Dataset có rung camera nhẹ, một số outlier trong training và một số mẫu bình thường xuất hiện ít.

Vai trò dự kiến:

- Dataset thứ hai để xác nhận pipeline và phương pháp.
- Có thể train lại cùng kiến trúc trên Avenue Train rồi đánh giá Avenue Test.
- Có thể dùng Avenue Test cho thí nghiệm cross-domain sau khi model đã hoạt động trên Ped2.

Hạn chế:

- Camera, nền, màu sắc và phân bố chuyển động khác Ped2.
- Outlier trong tập train có thể ảnh hưởng cách model học normality.
- Test trực tiếp bằng model train trên Ped2 dễ đo domain shift thay vì hành vi bất thường.

Nguồn:

- https://www.cse.cuhk.edu.hk/~leojia/projects/detectabnormal/dataset.html

## 6.3. Quyết định về giao thức hai dataset

Phương án khoa học và khả thi được khuyến nghị:

| Thí nghiệm | Training | Testing | Vai trò |
|---|---|---|---|
| Thí nghiệm chính | Ped2 Train | Ped2 Test | Phát triển và kiểm tra model |
| Xác nhận dataset thứ hai | Avenue Train | Avenue Test | Kiểm tra cùng phương pháp ở bối cảnh khác |
| Thí nghiệm nâng cao | Ped2 Train | Avenue Test | Đánh giá zero-shot cross-domain |

Không nên lấy Ped2 Train rồi Avenue Test làm thí nghiệm duy nhất vì:

- Nền, góc camera và màu ảnh thay đổi.
- Khái niệm bình thường khác nhau giữa hai scene.
- Model có thể gán score cao cho toàn bộ Avenue dù hành vi bình thường.
- Mục tiêu AUROC 90 đến 95 phần trăm trên Ped2 không còn ý nghĩa nếu không đánh giá Ped2 Test.

Nếu giảng viên bắt buộc **một dataset chỉ dùng training và một dataset chỉ dùng testing**, cần ghi rõ bài toán là:

> Zero-shot cross-domain video anomaly detection.

Trong trường hợp đó:

- Không nên cam kết AUROC 90 đến 95 phần trăm.
- Cần coi khả năng chịu domain shift là câu hỏi nghiên cứu chính.
- Nên bổ sung baseline rất đơn giản để biết score có đang phản ứng với scene mới hay không.
- Có thể đề xuất một thí nghiệm phụ dùng một lượng nhỏ video bình thường từ target để calibration, nhưng phải báo cáo riêng vì không còn là zero-shot tuyệt đối.

Vấn đề này chưa được xác nhận với giảng viên và là quyết định mở quan trọng nhất.

---

## 7. Tiền xử lý dữ liệu

Pipeline dữ liệu dự kiến:

1. Đọc video hoặc các frame ảnh.
2. Sắp xếp frame đúng thứ tự thời gian.
3. Resize và normalize.
4. Chuyển ảnh xám thành ba kênh nếu backbone pretrained yêu cầu RGB.
5. Tạo sliding window có độ dài ban đầu là 8 frame.
6. Gán target là frame hoặc feature của thời điểm kế tiếp.
7. Căn ground truth với target frame.
8. Trả về tensor có dạng `[batch, sequence, channel, height, width]`.

Các yêu cầu cần kiểm tra:

- Sequence không được vượt qua ranh giới giữa hai clip.
- Không được trộn frame từ train vào validation hoặc test.
- Thứ tự frame và ground truth phải khớp tuyệt đối.
- Resize không nên làm biến dạng tỷ lệ hình ảnh quá mức.
- Có thể thử 224 × 224 để đơn giản, nhưng phương án tốt hơn là resize kèm padding để giữ tỷ lệ. Kích thước cuối cần quyết định sau khi kiểm tra GPU.
- Frame sampling rate phải nhất quán giữa train và test.
- Augmentation không được tạo chuyển động phi thực tế hoặc phá nhãn thời gian.

Validation dự kiến:

- Giữ lại một phần clip bình thường từ training làm validation.
- Validation dùng để chọn checkpoint, ngưỡng cảnh báo và siêu tham số.
- Không dùng nhãn test để điều chỉnh threshold.

Kiểm tra bắt buộc trước khi train:

- Thống kê số clip và số frame.
- Kiểm tra độ phân giải, định dạng ảnh và số kênh.
- Hiển thị frame bình thường và bất thường.
- Overlay ground truth lên frame.
- Hiển thị một sequence hoàn chỉnh.
- Kiểm tra target frame của sequence.
- In shape và khoảng giá trị tensor.

---

## 8. Kiến trúc mô hình

## 8.1. Baseline bắt buộc: Conv-Autoencoder

Luồng xử lý:

```text
Frame -> CNN Encoder -> latent feature -> CNN Decoder -> reconstructed frame
```

Ý tưởng:

- Training chỉ dùng frame bình thường.
- Model tối thiểu hóa sai số tái tạo.
- Khi testing, sai số giữa frame gốc và frame tái tạo tạo thành anomaly score.

Vai trò:

- Kiểm tra data pipeline.
- Kiểm tra training loop, checkpoint và inference.
- Xây dựng cách tính score và evaluation.
- Tạo mốc so sánh với CNN-Transformer.

Ưu điểm:

- Dễ hiểu và triển khai.
- Huấn luyện nhẹ.
- Phù hợp để học quy trình PyTorch.

Hạn chế:

- Xử lý từng frame độc lập nên khó hiểu chuyển động.
- Autoencoder đôi khi vẫn tái tạo tốt vật thể bất thường.
- Reconstruction pixel có thể nhạy với ánh sáng và nhiễu camera.

Loss khởi đầu có thể dùng:

- L1 hoặc MSE reconstruction loss.
- Có thể thêm SSIM hoặc gradient loss sau khi baseline chạy ổn định.

Không nên bắt đầu bằng nhiều loss cùng lúc.

## 8.2. Mô hình chính: CNN và Temporal Transformer

Luồng đề xuất:

```text
Chuỗi 8 frame
-> CNN trích feature map cho từng frame
-> positional encoding
-> Temporal Transformer
-> dự đoán feature map của frame tiếp theo
-> feature prediction error
-> anomaly map và anomaly score
```

Cấu hình khởi đầu:

- Backbone: ResNet18 pretrained trên ImageNet.
- Trích feature tại layer3 hoặc một tầng còn giữ thông tin không gian.
- Sequence length: 8.
- Transformer Encoder: 2 lớp.
- Attention heads: 4.
- Embedding dimension: phụ thuộc feature CNN, có projection nếu cần.
- Prediction head: convolution hoặc linear projection phù hợp với dạng feature.
- Training chỉ dùng sequence bình thường.

Lý do giữ feature map không gian:

- Nếu global average pooling thành một vector cho mỗi frame, model mất nhiều thông tin vị trí.
- Một vector toàn frame khó tạo anomaly map.
- Decoder tái tạo ảnh từ vector đã pooling có thể hoạt động kém.

Phương án ưu tiên là **feature prediction** thay vì tái tạo pixel vì:

- Nhẹ hơn decoder tạo ảnh đầy đủ.
- Giảm ảnh hưởng của pixel noise và hiện tượng frame dự đoán bị mờ.
- Có thể tính sai số theo từng vị trí trên feature map rồi upsample thành anomaly map.

Anomaly score có thể thử:

- Trung bình sai số trên toàn feature map.
- Top-k trung bình của các vị trí có sai số cao nhất để tăng nhạy với bất thường nhỏ.
- Kết hợp L2 và cosine distance nếu cần.

Quyết định về hàm score cần được thực hiện bằng validation, không dựa vào test labels.

## 8.3. Mô hình mở rộng: ConvLSTM-Autoencoder

ConvLSTM có thể làm baseline thời gian không dùng attention. Tuy nhiên, mô hình này chỉ nên triển khai khi:

- Conv-Autoencoder chạy ổn định.
- CNN-Transformer đã train được.
- Vẫn còn thời gian cho việc đánh giá và demo.

Nếu tiến độ chậm, bỏ ConvLSTM không làm mất mục tiêu chính của dự án.

## 8.4. Reconstruction và prediction

Kế hoạch ban đầu ghi “reconstruction/prediction”, nhưng triển khai cần chọn một hướng chính.

Khuyến nghị:

- Conv-Autoencoder dùng reconstruction.
- CNN-Transformer dùng future feature prediction.
- Pixel prediction hoặc decoder ảnh là phần mở rộng.

Future prediction phù hợp với hệ thống online vì chỉ cần các frame quá khứ để đánh giá frame tiếp theo. Không nên dùng thông tin từ tương lai nếu muốn tuyên bố hệ thống hoạt động theo thời gian thực.

## 8.5. Attention map và khả năng giải thích

- Attention weight không tự động là lời giải thích đáng tin cậy cho quyết định model.
- Không nên gọi attention heatmap là bằng chứng model đã hiểu hành vi.
- Anomaly map từ sai số feature hoặc sai số prediction có liên hệ trực tiếp hơn với điểm bất thường.
- Nếu trình bày attention, cần ghi đây là trực quan hóa hỗ trợ, không phải ground truth giải thích.

---

## 9. Pipeline ứng dụng

Luồng xử lý dự kiến:

```text
Video file / webcam / RTSP
-> frame reader
-> sampling, resize, normalize
-> sliding buffer 8-16 frame
-> model inference
-> raw anomaly score
-> temporal smoothing
-> threshold and persistence rule
-> overlay, log and alert clip
```

### 9.1. Làm mượt và phát cảnh báo

Không nên cảnh báo ngay khi một frame đơn lẻ vượt ngưỡng. Các bước dự kiến:

1. Làm mượt score bằng moving average, exponential moving average hoặc median filter.
2. Dùng threshold được chọn trên validation.
3. Chỉ tạo sự kiện khi score vượt ngưỡng trong `K` frame liên tiếp.
4. Dùng hysteresis hoặc cooldown để tránh tạo nhiều cảnh báo cho cùng một sự kiện.
5. Lưu một số frame trước và sau thời điểm cảnh báo.

Các giá trị window, threshold và `K` chưa được chốt. Chúng phải được chọn từ validation và kiểm tra trên demo.

### 9.2. Đầu ra demo

Demo cuối kỳ nên có:

- Video đang phát.
- Anomaly score hiện tại.
- Đồ thị score theo thời gian.
- Trạng thái bình thường hoặc cảnh báo.
- Anomaly map hoặc vùng màu nếu model hỗ trợ.
- Danh sách sự kiện đã phát hiện.
- Khả năng mở lại ảnh hoặc clip của sự kiện.

Streamlit hoặc Gradio phù hợp cho bản demo học phần. Giao diện không cần phức tạp, nhưng pipeline inference và logging phải chạy thật.

### 9.3. Trường hợp lỗi cần dự kiến

- Camera rung hoặc đổi góc.
- Ánh sáng thay đổi đột ngột.
- Tín hiệu video bị mất hoặc frame lỗi.
- Người hoặc vật thể quá nhỏ, quá xa camera.
- Nền scene mới hoàn toàn khác training.
- Cảnh đông người làm feature thay đổi mạnh.

---

## 10. Đánh giá

## 10.1. Metric học thuật

- **Frame-level AUROC:** đo khả năng xếp frame bất thường cao hơn frame bình thường.
- **Average Precision hoặc AUPRC:** hữu ích khi frame bất thường chiếm tỷ lệ nhỏ.
- **Precision:** trong số cảnh báo, tỷ lệ cảnh báo đúng.
- **Recall:** trong số frame hoặc sự kiện bất thường, tỷ lệ được phát hiện.
- **F1-score:** cân bằng precision và recall tại một threshold cụ thể.
- **EER:** có thể báo cáo để đối chiếu kế hoạch ban đầu, nhưng không nên là metric chính cho tính ứng dụng.

AUROC không phải accuracy. Không được viết “độ chính xác 95%” nếu số đó thực chất là frame-level AUROC 0,95.

## 10.2. Metric hệ thống

- Số cảnh báo giả trên mỗi phút hoặc mỗi giờ video.
- Recall tại một mức false alarm rate cố định.
- Event-level precision, recall và F1 nếu có thể nhóm các frame thành sự kiện.
- Độ trễ từ khi sự kiện bắt đầu đến khi hệ thống cảnh báo.
- FPS khi inference.
- Latency cho mỗi frame hoặc mỗi window.
- GPU memory và RAM sử dụng nếu có điều kiện đo.

## 10.3. Nguyên tắc chống data leakage

- Không chọn threshold bằng test labels.
- Không tune hyperparameter trên Ped2 Test rồi báo cáo Ped2 Test như kết quả cuối không thiên lệch.
- Không normalize score bằng min và max của toàn bộ test video nếu hệ thống cần chạy online, vì cách này dùng thông tin tương lai.
- Không dùng test data để chọn epoch.
- Không để frame gần nhau từ cùng một clip rơi vào cả train và validation theo cách gây rò rỉ.
- Mọi preprocessing fit theo dữ liệu phải fit trên train hoặc validation, không fit trên test.

## 10.4. So sánh với literature

Kế hoạch ban đầu đề cập các mức AUROC tham khảo khoảng 90 đến trên 97 phần trăm. Các con số từ bài báo chỉ được dùng làm tài liệu tham khảo khi:

- Cùng dataset và cùng split.
- Cùng loại metric và cách tổng hợp score.
- Cùng quy trình normalization.
- Cùng kiểu supervision.
- Không nhầm kết quả same-domain với cross-domain.

Không nên đặt mục tiêu “so sánh được với SOTA” nếu chưa tái lập đúng protocol. Mục tiêu phù hợp hơn là:

- Baseline chạy đúng và tái lập được.
- CNN-Transformer cải thiện một cách có kiểm soát so với baseline hoặc cung cấp phân tích failure case rõ ràng.
- Demo có báo cáo false alarm và tốc độ.

---

## 11. Thiết kế thí nghiệm

### 11.1. Thứ tự thí nghiệm

1. Sanity check data loader.
2. Conv-Autoencoder trên một tập rất nhỏ để kiểm tra model có overfit được hay không.
3. Conv-Autoencoder đầy đủ trên Ped2.
4. Xây inference và tính anomaly score.
5. Đánh giá Ped2 Test.
6. CNN feature extractor và temporal prediction trên Ped2.
7. So sánh Conv-AE và CNN-Transformer.
8. Train và test theo split Avenue nếu thời gian cho phép.
9. Cross-domain Ped2 Train sang Avenue Test.
10. Demo video và đo tốc độ.

### 11.2. Ablation ưu tiên

Chỉ thực hiện các ablation trả lời câu hỏi rõ ràng:

| Ablation | Câu hỏi |
|---|---|
| Không Transformer và có Transformer | Transformer có giúp mô hình hóa thời gian không? |
| Sequence 8 và 16 | Chuỗi dài hơn có cải thiện hay chỉ tăng chi phí? |
| Mean score và top-k score | Cách tổng hợp nào nhạy hơn với bất thường nhỏ? |
| Feature prediction và pixel reconstruction | Loại mục tiêu nào ổn định hơn? |

Ablation thứ nhất là bắt buộc. Các ablation còn lại phụ thuộc tiến độ.

### 11.3. Bảng kết quả dự kiến

| Dataset | Model | AUROC | AP | F1 | False alarms/hour | FPS |
|---|---|---:|---:|---:|---:|---:|
| Ped2 | Conv-AE | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |
| Ped2 | CNN-Transformer | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |
| Avenue | Conv-AE | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |
| Avenue | CNN-Transformer | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |
| Ped2 to Avenue | CNN-Transformer | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |

Không điền kết quả tham khảo của bài báo vào bảng kết quả của dự án.

### 11.4. Experiment tracking

Mỗi lần chạy cần lưu:

- Tên dataset và split.
- Git commit.
- Random seed.
- Input resolution.
- Sequence length và sampling rate.
- Backbone và trạng thái frozen/unfrozen.
- Số layer, số head và embedding dimension.
- Loss và trọng số từng thành phần.
- Optimizer, learning rate, batch size và số epoch.
- Checkpoint tốt nhất theo validation.
- Metric test.
- FPS, latency và GPU memory nếu đo được.

Tên checkpoint nên rõ ràng, ví dụ:

```text
ped2_cnn_transformer_seq8_head4_seed42_best.pth
```

---

## 12. Lộ trình 8 tuần đã điều chỉnh

| Tuần | Trọng tâm | Đầu ra bắt buộc |
|---|---|---|
| 1 | Khảo sát bài toán | Phạm vi, dataset, mô hình, pipeline và rủi ro |
| 2 | Data pipeline | Dataset loader, thống kê, trực quan ground truth, sequence sampler |
| 3 | Baseline | Conv-Autoencoder train và inference được trên Ped2 |
| 4 | Temporal model | CNN feature extractor và Transformer nhỏ chạy được |
| 5 | Huấn luyện | Checkpoint chính, tuning tối thiểu bằng validation |
| 6 | Đánh giá | Metric Ped2, Avenue nếu có, ablation bắt buộc, failure cases |
| 7 | Hệ thống demo | Video input, score, smoothing, alert, logging, FPS |
| 8 | Hoàn thiện | Báo cáo, kiểm tra tái lập, chuẩn bị bảo vệ và buffer |

Nguyên tắc quản lý phạm vi:

- Conv-AE và CNN-Transformer là hai mô hình bắt buộc.
- ConvLSTM là phần mở rộng.
- Ped2 same-domain là đánh giá bắt buộc.
- Avenue same-domain là mục tiêu thứ hai.
- Cross-domain là thí nghiệm nâng cao.
- Demo video file là bắt buộc.
- Webcam hoặc RTSP là phần mở rộng nếu còn thời gian.

---

## 13. Sản phẩm dự kiến

### 13.1. Sản phẩm tối thiểu để dự án thành công

- Repository có cấu trúc rõ ràng.
- Data pipeline cho Ped2.
- Conv-Autoencoder baseline.
- CNN-Transformer hoặc CNN-Temporal Transformer.
- Checkpoint và config tương ứng.
- Kết quả Ped2 Test với AUROC, AP, F1 và failure cases.
- Demo nhận video file và hiển thị cảnh báo.
- Báo cáo mô tả phương pháp, thực nghiệm và giới hạn.

### 13.2. Sản phẩm mục tiêu

- Hỗ trợ cả Ped2 và Avenue.
- So sánh same-domain trên hai dataset.
- Một ablation chứng minh đóng góp của Transformer.
- Anomaly map hoặc heatmap sai số feature.
- Báo cáo false alarms/hour, FPS và latency.
- Lưu ảnh và clip khi cảnh báo.

### 13.3. Sản phẩm mở rộng

- ConvLSTM baseline.
- Cross-domain evaluation.
- Webcam hoặc RTSP.
- Calibration bằng video bình thường của camera mục tiêu.
- Giao diện Streamlit hoặc Gradio hoàn chỉnh hơn.

---

## 14. Cấu trúc repository dự kiến

Kế hoạch thiết lập ban đầu đề xuất:

```text
Video-anomaly-detection/
├── data/                         # không commit vào Git
├── src/
│   ├── preprocessing.py
│   ├── datasets/
│   │   ├── ped2.py
│   │   └── avenue.py
│   ├── models/
│   │   ├── conv_ae.py
│   │   ├── convlstm_ae.py        # tùy chọn
│   │   └── cnn_transformer.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── configs/
├── notebooks/
├── demo/
├── report/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

`.gitignore` tối thiểu:

```gitignore
data/
checkpoints/
*.pth
*.ckpt
__pycache__/
.ipynb_checkpoints/
venv/
.venv/
*.pyc
.env
```

`requirements.txt` ban đầu:

```text
torch
torchvision
opencv-python
numpy
matplotlib
scikit-learn
tqdm
```

Có thể bổ sung sau:

- `pyyaml` cho config.
- `pandas` cho bảng kết quả.
- `tensorboard` hoặc `wandb` cho theo dõi thí nghiệm.
- `streamlit` hoặc `gradio` cho demo.
- `scikit-image` nếu cần SSIM.

Không nên thêm toàn bộ thư viện trước khi thực sự sử dụng.

---

## 15. Quy trình làm việc local, GitHub và cloud

Luồng dự kiến:

```text
Local development
-> Git commit and push
-> Colab or Kaggle clone/pull
-> GPU training
-> checkpoint saved to Drive or Kaggle storage
-> result files committed or downloaded for analysis
```

Nguyên tắc:

- Code và config lưu trong Git.
- Dataset và checkpoint không lưu trong Git.
- Kết quả nhẹ như CSV, JSON, hình biểu đồ có thể commit.
- Không đưa GitHub Personal Access Token vào notebook đã commit.
- Colab cần dùng PAT hoặc credential phù hợp nếu push trực tiếp.
- Backup checkpoint sau mỗi thí nghiệm quan trọng vì phiên Colab có thể bị ngắt.

Repository GitHub đã được cấu hình từ trước:

```text
https://github.com/ktlletrungkien-png/Video-anomaly-detection.git
```

Tại thời điểm tổng hợp tài liệu này, repository local chưa có commit trên nhánh `main`.

---

## 16. Rủi ro và biện pháp xử lý

| Rủi ro | Hậu quả | Biện pháp |
|---|---|---|
| Chưa có nền tảng AI | Mất thời gian sửa model phức tạp | Học theo từng phần, hoàn thành baseline trước |
| Phạm vi quá rộng | Không hoàn thành hệ thống | Giới hạn camera cố định và bất thường về đối tượng/chuyển động |
| Ba mô hình trong 8 tuần | Không đủ thời gian đánh giá và demo | ConvLSTM trở thành tùy chọn |
| Dataset nhỏ | Overfitting | Validation theo clip, augmentation có kiểm soát, backbone pretrained |
| Domain shift giữa Ped2 và Avenue | Score phản ánh scene thay vì hành vi | Same-domain trước, cross-domain sau |
| Global pooling mất vị trí | Không tạo được anomaly map tốt | Giữ feature map không gian |
| Pixel reconstruction bị mờ | Score không ổn định | Ưu tiên feature prediction cho mô hình chính |
| Autoencoder tái tạo được anomaly | Chênh lệch score nhỏ | Dùng temporal prediction, feature error hoặc top-k score |
| Rung camera và đổi ánh sáng | Nhiều false positive | Augmentation, smoothing, camera motion check |
| Sequence hoặc nhãn lệch | Metric sai hoàn toàn | Visual sanity check bắt buộc |
| Tune trên test | Kết quả bị leakage | Validation riêng, khóa test đến cuối |
| Per-video min-max normalization | Không phù hợp online, dùng thông tin tương lai | Calibration từ train/validation |
| AUROC cao nhưng báo giả nhiều | Demo không dùng được | Báo cáo AP, F1, false alarms/hour và event metrics |
| Transformer nặng | Hết GPU/RAM | ResNet18, sequence 8, 2 layers, mixed precision nếu ổn định |
| Colab ngắt phiên | Mất checkpoint | Lưu checkpoint lên Drive |
| Dataset/checkpoint vào Git | Repo nặng hoặc bị từ chối push | `.gitignore`, Drive hoặc Kaggle storage |

---

## 17. Tiêu chí thành công đã điều chỉnh

Không dùng “đạt SOTA” làm tiêu chí bắt buộc. Dự án được xem là thành công khi:

1. Data pipeline đúng và kiểm chứng bằng trực quan.
2. Baseline train, lưu checkpoint và inference được.
3. Mô hình chính xử lý chuỗi thời gian và tạo anomaly score.
4. Có so sánh công bằng giữa baseline và mô hình chính.
5. Có đánh giá Ped2 theo split chính thức.
6. Có demo video hoàn chỉnh từ input đến cảnh báo.
7. Báo cáo cả kết quả tốt, failure case, false alarm và tốc độ.
8. Code và config đủ để chạy lại thí nghiệm.

Mục tiêu frame-level AUROC 90 đến 95 phần trăm trên Ped2 có thể giữ làm mục tiêu tham khảo, không phải cam kết. Không áp dụng mục tiêu này cho thí nghiệm Ped2 sang Avenue.

---

## 18. Kết quả khảo sát tuần 1

Các nội dung đã hoàn thành ở mức phân tích và thiết kế:

- Đã đọc tài liệu thiết lập môi trường `SETUP.md`.
- Đã đọc và đánh giá kế hoạch `PROJECT_PLAN.md`.
- Đã xác định kế hoạch ban đầu hợp lý cho proof of concept nhưng chưa đủ cho hệ thống ứng dụng.
- Đã thu hẹp phạm vi về camera cố định và khu vực dành cho người đi bộ.
- Đã giải thích thiết lập normal-only learning.
- Đã khảo sát vai trò của Ped2 và Avenue.
- Đã xác định rủi ro của việc train Ped2 rồi test Avenue.
- Đã đề xuất giao thức same-domain và cross-domain riêng biệt.
- Đã chọn Conv-Autoencoder làm baseline bắt buộc.
- Đã đề xuất CNN-Temporal Transformer dùng future feature prediction làm mô hình chính.
- Đã xác định pipeline ứng dụng từ video đến cảnh báo và lưu sự kiện.
- Đã bổ sung metric vận hành bên cạnh AUROC.
- Đã điều chỉnh lộ trình 8 tuần phù hợp với người biết lập trình nhưng mới học AI.
- Đã lập danh sách rủi ro và biện pháp giảm thiểu.

Những nội dung **chưa thực hiện**:

- Chưa tạo đầy đủ cấu trúc source code.
- Chưa tải hoặc kiểm tra trực tiếp Ped2 và Avenue trong repository.
- Chưa xây Dataset hoặc DataLoader.
- Chưa xây model.
- Chưa chạy training.
- Chưa có checkpoint.
- Chưa có kết quả AUROC, AP, F1, false alarms/hour hoặc FPS.
- Chưa có demo inference.
- Chưa xác nhận với giảng viên cách hiểu yêu cầu “một dataset train, một dataset test”.

---

## 19. Công việc ưu tiên tiếp theo

### Việc 1. Xác nhận yêu cầu dataset

Hỏi giảng viên:

> Yêu cầu hai dataset có bắt buộc model chỉ được train trên dataset thứ nhất và không được dùng training split của dataset thứ hai, hay có thể đánh giá mỗi dataset theo train/test split chính thức rồi thực hiện thêm cross-dataset test?

Đây là câu hỏi ảnh hưởng trực tiếp tới mục tiêu, metric và khả năng đạt kết quả.

### Việc 2. Khởi tạo cấu trúc repository

- Tạo `src`, `configs`, `notebooks`, `demo`, `tests` và các file nền.
- Tạo `.gitignore` và `requirements.txt`.
- Viết README mô tả mục tiêu và cách chạy.
- Tạo commit đầu tiên.

### Việc 3. Tải và kiểm tra dataset

- Tải Ped2 trước.
- Xác nhận train/test, số clip, số frame và ground truth.
- Hiển thị một số frame và mask.
- Sau khi Ped2 pipeline ổn định mới thêm Avenue.

### Việc 4. Xây data pipeline

- Implement frame reader.
- Implement sequence sampler.
- Implement label alignment.
- Viết test chống vượt ranh giới clip.
- Kiểm tra một batch.

### Việc 5. Xây baseline

- Conv-Autoencoder nhỏ.
- Sanity check overfit một batch.
- Train đầy đủ Ped2.
- Lưu reconstruction, anomaly score và checkpoint.

### Việc 6. Hoàn thiện evaluation trước mô hình chính

- AUROC và AP.
- Threshold từ validation.
- Precision, recall, F1.
- Timeline score.
- False positive examples.

Chỉ sau các bước này mới bắt đầu CNN-Transformer.

---

## 20. Câu hỏi có thể gặp khi bảo vệ

### Vì sao chỉ train bằng dữ liệu bình thường?

Sự kiện bất thường hiếm, đa dạng và khó thu thập đầy đủ. Học normality giúp model có thể gán score cao cho các mẫu khác với quy luật đã học, kể cả khi loại bất thường đó chưa xuất hiện trong training.

### Vì sao chọn Ped2?

Ped2 nhỏ, split rõ ràng và có ground truth theo frame. Dataset phù hợp để xây proof of concept trong thời gian giới hạn.

### Vì sao cần Avenue?

Avenue đa dạng hơn và có vùng ground truth. Dataset giúp kiểm tra cùng phương pháp trong một bối cảnh khác và hỗ trợ phân tích khả năng tổng quát hóa.

### Có thể train Ped2 rồi test Avenue không?

Có thể, nhưng đó là cross-domain evaluation. Camera và phân bố dữ liệu khác nhau nên model có thể phản ứng với scene mới thay vì hành vi bất thường. Kết quả phải được báo cáo riêng.

### Vì sao dùng Transformer?

CNN xử lý thông tin không gian trong từng frame. Transformer học quan hệ giữa nhiều frame liên tiếp, từ đó có thể phát hiện chuyển động không phù hợp với quy luật đã học.

### Làm sao chứng minh Transformer có ích?

So sánh hai cấu hình giống nhau, chỉ thay đổi việc có hoặc không có Transformer. Đây là ablation bắt buộc.

### Làm sao đặt ngưỡng cảnh báo?

Dùng phân bố anomaly score của validation bình thường. Không dùng test labels để chọn ngưỡng.

### AUROC cao có đủ để ứng dụng không?

Không. Cần đánh giá thêm AP, precision, recall, F1, số cảnh báo giả, độ trễ và FPS.

### Hệ thống có chạy real-time không?

Chưa thể cam kết trước khi đo. Kiến trúc nhẹ được chọn để hướng tới gần thời gian thực. Kết quả cuối phải báo cáo FPS và latency trên thiết bị cụ thể.

### Hệ thống có phát hiện được đánh nhau hoặc trộm cắp không?

Không nên khẳng định với Ped2. Phạm vi hiện tại là bất thường về đối tượng và chuyển động trong camera cố định. Các hành vi có ngữ nghĩa phức tạp cần dataset và phương pháp khác.

---

## 21. Tài liệu tham khảo chính

1. Mahadevan, V., Li, W., Bhalodia, V., Vasconcelos, N. “Anomaly Detection in Crowded Scenes.” CVPR 2010.  
   https://www.svcl.ucsd.edu/projects/anomaly/

2. Li, W., Mahadevan, V., Vasconcelos, N. “Anomaly Detection and Localization in Crowded Scenes.” IEEE TPAMI, 2014.  
   https://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf

3. Lu, C., Shi, J., Jia, J. “Abnormal Event Detection at 150 FPS in MATLAB.” ICCV 2013.  
   https://www.cse.cuhk.edu.hk/~leojia/projects/detectabnormal/dataset.html

4. Liu, W., Luo, W., Lian, D., Gao, S. “Future Frame Prediction for Anomaly Detection: A New Baseline.” CVPR 2018.  
   https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html

5. Vaswani, A. et al. “Attention Is All You Need.” NeurIPS 2017.  
   https://arxiv.org/abs/1706.03762

6. Gong, D. et al. “Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection.” ICCV 2019.  
   https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html

7. Aich, A., Peng, K., Roy-Chowdhury, A. “Cross-Domain Video Anomaly Detection Without Target Domain Adaptation.” WACV 2023.  
   https://openaccess.thecvf.com/content/WACV2023/papers/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.pdf

8. Ramachandra, B., Jones, M., Vatsavai, R. “Street Scene: A New Dataset and Evaluation Protocol for Video Anomaly Detection.” WACV 2020.  
   https://openaccess.thecvf.com/content_WACV_2020/html/Ramachandra_Street_Scene_A_new_dataset_and_evaluation_protocol_for_video_WACV_2020_paper.html

---

## 22. Quyết định của kế hoạch Ped2/Avenue trước đây, chỉ giữ để tham chiếu

### Đã chốt ở mức kế hoạch

- Giữ chủ đề Video Anomaly Detection.
- Thu hẹp vào camera cố định và bất thường về đối tượng/chuyển động.
- PyTorch là framework chính.
- Colab Pro hoặc Kaggle GPU dùng để huấn luyện.
- Ped2 là dataset phát triển chính.
- Avenue là dataset thứ hai.
- Conv-Autoencoder là baseline bắt buộc.
- CNN-Temporal Transformer là mô hình chính.
- Ưu tiên future feature prediction.
- Demo phải có score, cảnh báo và lưu sự kiện.
- Metric phải gồm cả AUROC/AP và chỉ số vận hành.

### Còn mở

- Giảng viên có bắt buộc train trên một dataset và test duy nhất trên dataset khác hay không.
- Input resolution cuối cùng.
- Sampling rate và sequence length cuối cùng.
- Frozen hoặc fine-tune những tầng nào của ResNet18.
- Loss cuối cùng của feature prediction.
- Cách tổng hợp anomaly score.
- Cách chọn threshold và số frame liên tiếp `K`.
- Có đủ thời gian làm ConvLSTM hay không.
- Demo chỉ dùng video file hay có thêm webcam/RTSP.
- Có train và đánh giá Avenue same-domain hay chỉ làm cross-domain.

---

## 23. Hướng dẫn sử dụng tài liệu này cho các lần làm việc sau

Tài liệu này là nguồn bối cảnh chính. Trước khi triển khai bước mới cần:

1. Đọc **phần 0** trước vì đây là hướng phát hiện bạo lực đang có hiệu lực.
2. Không giả định dataset, model hoặc kết quả đã tồn tại.
3. Không mở rộng phạm vi sang nhận dạng khuôn mặt hoặc nhiều loại tội phạm nếu chưa có yêu cầu mới.
4. Ưu tiên các bước trong mục 0.12, bắt đầu từ xác nhận protocol dataset và xây RWF-2000 data pipeline.
5. Cập nhật tài liệu khi có quyết định mới, kết quả mới hoặc thay đổi phạm vi.
6. Ghi rõ giá trị nào là kết quả đã đo và giá trị nào chỉ là mục tiêu tham khảo.

Trạng thái hiện tại có thể tóm tắt bằng một câu:

> Dự án đã chốt hệ thống lai: RWF-2000 cho nhánh phát hiện bạo lực, Ped2 và Avenue cho nhánh bất thường tổng quát; UBI-Fights là external test tùy tiến độ. Dự án chưa bắt đầu xây dựng data pipeline hoặc mô hình AI.
