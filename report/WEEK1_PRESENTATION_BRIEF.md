# Slide brief tuần 1

## Hệ thống phát hiện hành vi bất thường trong video CCTV

Tài liệu này là nguồn đầu vào cho plugin tạo presentation. Nội dung trong phần **Chữ trên slide** dùng để hiển thị trực tiếp. Nội dung trong phần **Ghi chú thuyết trình** phải đưa vào speaker notes, không đặt toàn bộ lên slide.

---

## 1. Yêu cầu chung cho plugin presentation

### Đầu ra

- Tạo bài thuyết trình bằng tiếng Việt, tỷ lệ 16:9.
- Tổng cộng 15 slide, tính cả trang bìa.
- Thời lượng trình bày dự kiến: 10 đến 15 phút.
- Đối tượng nghe: giảng viên và sinh viên học phần Thiết kế hệ thống thông minh.
- Mục tiêu của bài trình bày: báo cáo kết quả tìm hiểu đề tài trong tuần 1, chưa trình bày kết quả huấn luyện.
- Giữ bảng và sơ đồ dưới dạng đối tượng có thể chỉnh sửa.
- Đưa nguồn tài liệu vào speaker notes của slide liên quan.
- Không tự tạo số liệu thực nghiệm, accuracy, AUC hoặc trạng thái công việc chưa được xác nhận.

### Phong cách hình ảnh

- Phong cách học thuật, hiện đại, nghiêm túc.
- Nền xanh navy đậm hoặc trắng ngà. Dùng một phong cách nhất quán trong toàn bộ deck.
- Màu chính gợi ý: navy `#0B1F33`, xanh teal `#159E9C`, trắng ngà `#F6F7F8`.
- Chỉ dùng màu cam `#F59E0B` hoặc đỏ `#DC4C4C` để biểu thị cảnh báo bất thường.
- Dùng một họ font hỗ trợ tiếng Việt, ưu tiên Aptos, Arial hoặc Be Vietnam Pro.
- Tiêu đề slide tối thiểu 30 pt. Nội dung tối thiểu 18 pt.
- Tránh bố cục dạng dashboard với nhiều thẻ nhỏ. Mỗi slide nên có một bố cục chính rõ ràng.
- Dùng ảnh CCTV hoặc frame chính thức từ dataset khi cần minh họa. Không dùng ảnh AI giả làm dữ liệu thực nghiệm.
- Không dùng hình nhận diện khuôn mặt vì đề tài không xử lý danh tính.
- Không lặp lại cùng một ảnh minh họa ở nhiều slide.

### Quy tắc nội dung

- Tiêu đề ngắn, gọi đúng tên nội dung của slide.
- Mỗi slide chỉ có một thông điệp chính.
- Không dùng các câu quảng cáo như “giải pháp đột phá” hoặc “công nghệ tiên tiến”.
- Không tuyên bố hệ thống phát hiện được mọi hành vi đáng ngờ.
- Dùng thuật ngữ `frame-level AUROC`, không gọi AUROC là accuracy.
- Phân biệt rõ kết quả tham khảo trong bài báo và kết quả của dự án.
- Chưa có kết quả huấn luyện thì hiển thị “Chưa thực hiện”, không điền số giả định.

### Thông tin cần người dùng điền trước khi xuất bản

- `[TÊN SINH VIÊN/NHÓM]`
- `[MÃ SỐ SINH VIÊN]`
- `[TÊN GIẢNG VIÊN]`
- `[LỚP/HỌC PHẦN]`
- `[NGÀY BÁO CÁO]`
- Trạng thái thực tế của các đầu việc ở slide 13.

---

## 2. Nội dung chi tiết từng slide

## Slide 1. Trang bìa

### Mục đích

Giới thiệu đề tài và xác định đây là báo cáo khảo sát tuần 1.

### Chữ trên slide

**HỆ THỐNG PHÁT HIỆN HÀNH VI BẤT THƯỜNG TRONG VIDEO CCTV**

Khảo sát bài toán và thiết kế sơ bộ

Học phần: Thiết kế hệ thống thông minh

`[TÊN SINH VIÊN/NHÓM]`  
`[MÃ SỐ SINH VIÊN]`  
`[TÊN GIẢNG VIÊN]`  
`[NGÀY BÁO CÁO]`

### Bố cục và hình ảnh

- Bìa tối giản.
- Đặt tiêu đề ở nửa trái.
- Nửa phải dùng một ảnh CCTV góc cao nhìn xuống lối đi bộ hoặc hành lang.
- Phủ lớp màu navy trong suốt lên ảnh để chữ dễ đọc.
- Không đặt sơ đồ kiến trúc hoặc nhiều biểu tượng trên trang bìa.

### Ghi chú thuyết trình

Đây là báo cáo kết quả tìm hiểu trong tuần đầu của dự án. Đề tài hướng tới một hệ thống hỗ trợ nhân viên giám sát bằng cách tự động tính mức độ bất thường của từng đoạn video và phát cảnh báo khi phát hiện đối tượng hoặc chuyển động khác với bối cảnh bình thường. Trong tuần đầu, trọng tâm là xác định bài toán, khảo sát dữ liệu, lựa chọn hướng mô hình và thiết kế sơ bộ hệ thống. Dự án chưa bước vào giai đoạn huấn luyện nên bài trình bày chưa đưa ra số liệu thực nghiệm.

---

## Slide 2. Bối cảnh của bài toán

### Mục đích

Giải thích nhu cầu thực tế và vai trò hỗ trợ của hệ thống.

### Chữ trên slide

**Bối cảnh giám sát video**

- Camera tạo ra lượng video lớn và hoạt động liên tục.
- Người vận hành khó theo dõi đồng thời nhiều luồng hình ảnh.
- Sự kiện bất thường xuất hiện ít nhưng thường cần phản ứng sớm.
- Hệ thống AI có thể sàng lọc video và đề xuất thời điểm cần kiểm tra.

**Vai trò dự kiến**

Hỗ trợ cảnh báo cho người vận hành. Con người vẫn xác minh sự kiện và đưa ra quyết định.

### Bố cục và hình ảnh

- Dùng một ảnh toàn cảnh phòng giám sát hoặc nhiều màn hình CCTV ở bên trái.
- Bên phải đặt bốn ý chính, không đặt trong bốn thẻ riêng biệt.
- Dòng “Vai trò dự kiến” đặt ở chân slide và tô màu teal.

### Gợi ý tìm ảnh

- Từ khóa: `CCTV monitoring room surveillance screens operator`.
- Tránh ảnh có logo công ty hoặc khuôn mặt nổi bật.

### Ghi chú thuyết trình

Mục tiêu của đề tài không phải thay thế nhân viên giám sát. Hệ thống đóng vai trò bộ lọc, giúp xác định đoạn video có dấu hiệu khác thường để con người kiểm tra. Trong thực tế, một người không thể quan sát tập trung nhiều camera trong thời gian dài. Tuy nhiên, việc phát cảnh báo sai quá nhiều cũng làm hệ thống mất giá trị. Vì vậy, ngoài chỉ số trên dataset, dự án cần quan tâm đến số cảnh báo giả, độ trễ và tốc độ xử lý.

---

## Slide 3. Phát biểu bài toán và phạm vi

### Mục đích

Xác định rõ đầu vào, nhiệm vụ, đầu ra và giới hạn của phiên bản đầu.

### Chữ trên slide

**Phát biểu bài toán**

Học quy luật hoạt động bình thường từ video camera cố định, sau đó phát hiện frame hoặc đoạn thời gian có đối tượng và chuyển động khác với quy luật đã học.

**Đầu vào**

- Video có sẵn, webcam hoặc luồng RTSP.
- Chuỗi 8 đến 16 frame liên tiếp.

**Đầu ra**

- Anomaly score cho từng frame.
- Trạng thái bình thường hoặc cảnh báo.
- Thời gian và hình ảnh của sự kiện nghi ngờ.

**Phạm vi phiên bản đầu**

Camera cố định tại khu vực dành cho người đi bộ. Không nhận dạng khuôn mặt, danh tính hoặc ý định của con người.

### Bố cục và hình ảnh

- Dùng sơ đồ ngang có ba vùng: Đầu vào, Xử lý, Đầu ra.
- Dùng đường nối thẳng và biểu tượng đơn giản.
- Đặt phạm vi thành một dòng riêng ở cuối slide.
- Giữ sơ đồ có thể chỉnh sửa.

### Ghi chú thuyết trình

Khái niệm bất thường phụ thuộc vào bối cảnh. Xe đạp là bình thường trên đường giao thông nhưng có thể là bất thường trong lối đi chỉ dành cho người đi bộ. Vì vậy, dự án giới hạn bối cảnh ở camera cố định và tập trung vào bất thường về đối tượng hoặc chuyển động. Phiên bản đầu chưa giải quyết các khái niệm mang tính ý định như trộm cắp hoặc hành vi đáng ngờ. Đầu ra chính là một điểm số theo frame, sau đó hệ thống áp dụng quy tắc thời gian để quyết định cảnh báo.

---

## Slide 4. Cách học từ dữ liệu bình thường

### Mục đích

Giải thích normal-only learning bằng sơ đồ trực quan cho người chưa học sâu về AI.

### Chữ trên slide

**Giai đoạn huấn luyện**

Video bình thường được dùng để mô hình học hình ảnh và chuyển động thường gặp.

**Giai đoạn kiểm tra**

Video mới chứa cả frame bình thường và bất thường. Frame khác nhiều so với quy luật đã học sẽ nhận anomaly score cao.

**Vai trò của nhãn**

Nhãn bất thường không tham gia huấn luyện. Nhãn chỉ dùng để đánh giá kết quả trên tập test.

### Bố cục và hình ảnh

- Chia slide thành hai nửa theo chiều ngang hoặc chiều dọc.
- Nửa đầu là luồng Training: frame người đi bộ, model, quy luật bình thường.
- Nửa sau là luồng Testing: frame mới, anomaly score, cảnh báo.
- Dùng màu teal cho bình thường và cam cho bất thường.
- Tạo sơ đồ bằng đối tượng native, không biến sơ đồ thành ảnh phẳng.

### Ghi chú thuyết trình

Trong nhiều bài báo, cách thiết lập này được gọi là unsupervised video anomaly detection. Một tên mô tả chính xác hơn là one-class hoặc normal-only learning vì quá trình huấn luyện biết trước rằng video train chứa hành vi bình thường. Cách tiếp cận này phù hợp khi bất thường hiếm và không thể liệt kê đầy đủ. Model không học tên của từng hành vi xấu. Model học mẫu bình thường, sau đó đo mức độ sai khác của dữ liệu mới.

---

## Slide 5. Dataset UCSD Ped2

### Mục đích

Giới thiệu dataset chính và lý do dataset phù hợp với giai đoạn phát triển.

### Chữ trên slide

**UCSD Ped2**

- Bối cảnh: lối đi bộ, camera cố định đặt ở vị trí cao.
- Kích thước ảnh: khoảng 360 × 240 pixel.
- Training: 16 clip chỉ chứa hoạt động bình thường.
- Testing: 12 clip chứa cả bình thường và bất thường.
- Bất thường thường gặp: xe đạp, skateboard, xe nhỏ và chuyển động khác quy luật.

**Vai trò trong dự án**

Dataset phát triển chính vì dung lượng nhỏ, ground truth rõ ràng và phù hợp huấn luyện trên Colab hoặc Kaggle.

Nguồn: UCSD Statistical Visual Computing Lab.

### Bố cục và hình ảnh

- Bên trái dùng hai frame chính thức: một frame bình thường và một frame bất thường.
- Bên phải đặt thông tin dataset.
- Đánh dấu vật thể bất thường bằng khung màu cam nếu ảnh nguồn có ground truth.
- Ghi chú rõ “Ảnh minh họa từ dataset”, không trình bày như kết quả của model.

### Ghi chú thuyết trình

Ped2 là lựa chọn phù hợp để xây dựng proof of concept vì dataset tương đối nhỏ và bối cảnh rõ ràng. Model có thể học rằng hoạt động bình thường chủ yếu là người đi bộ theo một số hướng quen thuộc. Xe đạp, skateboard hoặc xe nhỏ tạo ra sự khác biệt về hình dáng và chuyển động. Hạn chế quan trọng là Ped2 chỉ đại diện cho một cảnh cụ thể. Kết quả tốt trên Ped2 chưa đủ chứng minh mô hình có thể triển khai trên mọi camera.

### Nguồn cho speaker notes

- https://www.svcl.ucsd.edu/projects/anomaly/
- https://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf

---

## Slide 6. Dataset CUHK Avenue

### Mục đích

Giới thiệu dataset thứ hai và các thách thức mới so với Ped2.

### Chữ trên slide

**CUHK Avenue**

- Bối cảnh: lối đi trong khuôn viên trường, video màu.
- Tổng số: 30.652 frame.
- Training: 16 video chứa tình huống bình thường.
- Testing: 21 video chứa cả bình thường và bất thường.
- Ground truth gồm thời gian và vùng xảy ra sự kiện.

**Thách thức**

Dataset có rung camera, một số outlier trong tập train và các mẫu bình thường xuất hiện ít.

Nguồn: CUHK Avenue Dataset.

### Bố cục và hình ảnh

- Dùng một frame chính thức có vùng bất thường ở nửa phải.
- Thông tin dataset đặt ở nửa trái.
- Thể hiện số 16 và 21 nổi bật nhưng không dùng kiểu dashboard.

### Ghi chú thuyết trình

Avenue đa dạng hơn Ped2 và cung cấp bounding box cho vùng bất thường. Dataset phù hợp để kiểm tra liệu cùng một pipeline có hoạt động trong bối cảnh khác hay không. Tuy nhiên, Avenue và Ped2 khác về camera, màu sắc, nền và loại hành vi. Nếu dùng model train trên Ped2 để test trực tiếp trên Avenue, model có thể phản ứng với sự thay đổi bối cảnh thay vì hành vi bất thường.

### Nguồn cho speaker notes

- https://www.cse.cuhk.edu.hk/~leojia/projects/detectabnormal/dataset.html

---

## Slide 7. Chiến lược sử dụng hai dataset

### Mục đích

Trình bày giao thức đánh giá chính và cross-dataset test như một thí nghiệm nâng cao.

### Chữ trên slide

**Chiến lược thực nghiệm đề xuất**

| Thí nghiệm | Dữ liệu huấn luyện | Dữ liệu kiểm tra | Mục đích |
|---|---|---|---|
| Thí nghiệm chính | Ped2 Train | Ped2 Test | Phát triển và kiểm tra mô hình |
| Xác nhận dataset thứ hai | Avenue Train | Avenue Test | Kiểm tra tính ổn định của phương pháp |
| Thí nghiệm nâng cao | Ped2 Train | Avenue Test | Đánh giá khả năng tổng quát hóa chéo miền |

**Rủi ro cross-domain**

Khác biệt camera và bối cảnh có thể tạo anomaly score cao ngay cả khi hành vi vẫn bình thường.

### Bố cục và hình ảnh

- Bảng chiếm phần lớn slide và phải có thể chỉnh sửa.
- Tô nền nhạt cho hàng “Thí nghiệm chính”.
- Tô màu cam nhạt cho hàng “Thí nghiệm nâng cao”.
- Đặt cảnh báo cross-domain thành một dòng ở chân slide.

### Ghi chú thuyết trình

Giao thức chính nên tuân theo split chính thức của từng dataset. Ped2 Train được dùng để huấn luyện và Ped2 Test dùng để đánh giá. Sau khi pipeline ổn định, cùng mã nguồn được train lại trên Avenue Train và đánh giá trên Avenue Test. Việc train Ped2 rồi test Avenue vẫn có giá trị nhưng cần được gọi đúng là cross-domain hoặc zero-shot cross-domain evaluation. Nếu giảng viên bắt buộc một dataset chỉ dùng train và dataset còn lại chỉ dùng test, dự án phải xác định đây là câu hỏi nghiên cứu chính và chấp nhận rằng điểm số có thể giảm mạnh.

### Nguồn cho speaker notes

- Cross-Domain Video Anomaly Detection Without Target Domain Adaptation, WACV 2023.
- https://openaccess.thecvf.com/content/WACV2023/papers/Aich_Cross-Domain_Video_Anomaly_Detection_Without_Target_Domain_Adaptation_WACV_2023_paper.pdf

---

## Slide 8. Baseline Conv-Autoencoder

### Mục đích

Giải thích mô hình khởi đầu và cách tạo anomaly score.

### Chữ trên slide

**Conv-Autoencoder**

1. CNN Encoder nén frame thành feature.
2. CNN Decoder tái tạo lại frame đầu vào.
3. Sai số giữa frame gốc và frame tái tạo tạo thành anomaly score.

**Giả thuyết**

Model học tốt các frame bình thường. Frame bất thường khó tái tạo hơn nên có sai số cao hơn.

**Vai trò**

Baseline đơn giản để kiểm tra data pipeline, training loop và cách tính điểm bất thường.

### Bố cục và hình ảnh

- Sơ đồ ngang có bốn phần: Frame đầu vào, Encoder, Feature, Decoder và Frame tái tạo.
- Dùng một vùng sai khác giữa frame gốc và frame tái tạo để minh họa anomaly map.
- Sơ đồ cần có thể chỉnh sửa.
- Không dùng quá nhiều công thức.

### Ghi chú thuyết trình

Conv-Autoencoder là mô hình phù hợp cho người mới vì có cấu trúc rõ ràng. Encoder giảm kích thước ảnh và rút ra đặc trưng. Decoder cố gắng khôi phục ảnh ban đầu. Khi huấn luyện chỉ bằng ảnh bình thường, model được kỳ vọng sẽ tái tạo tốt các mẫu quen thuộc. Tuy nhiên, autoencoder đôi khi vẫn tái tạo được vật thể bất thường, và việc xử lý từng frame riêng lẻ làm model khó hiểu chuyển động. Vì vậy, Conv-Autoencoder đóng vai trò baseline thay vì mô hình cuối cùng.

---

## Slide 9. Mô hình CNN và Temporal Transformer

### Mục đích

Trình bày mô hình chính và vai trò của từng khối.

### Chữ trên slide

**Kiến trúc mô hình đề xuất**

Chuỗi 8 frame được đưa qua CNN để tạo feature map. Temporal Transformer phân tích sự thay đổi theo thời gian và dự đoán feature của frame tiếp theo.

**Các khối chính**

- ResNet18 pretrained: trích xuất đặc trưng không gian.
- Positional encoding: giữ thông tin thứ tự frame.
- Transformer Encoder 2 lớp: học quan hệ thời gian.
- Prediction head: dự đoán feature của frame tiếp theo.
- Feature error: tạo anomaly score và anomaly map.

### Bố cục và hình ảnh

- Sơ đồ kiến trúc chạy từ trái sang phải.
- Dùng dải 8 thumbnail nhỏ để biểu diễn chuỗi frame.
- Sau CNN, biểu diễn feature map bằng một khối lưới đơn giản.
- Sau Transformer, hiển thị feature dự đoán và feature thực tế.
- Đánh dấu phép so sánh feature bằng màu cam.
- Sơ đồ phải là native diagram có thể chỉnh sửa.

### Ghi chú thuyết trình

CNN xử lý nội dung trong từng frame, còn Transformer mô hình hóa quan hệ giữa nhiều thời điểm. Phiên bản đầu sử dụng ResNet18 vì nhẹ hơn các backbone lớn. Nên giữ feature map không gian thay vì chỉ dùng một vector sau global average pooling, vì thông tin vị trí cần thiết để tạo anomaly map. Dự đoán feature tiếp theo nhẹ và ổn định hơn việc tái tạo toàn bộ ảnh. Nếu thời gian cho phép, dự án có thể bổ sung decoder để tạo ảnh dự đoán trực quan.

### Nguồn cho speaker notes

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Future Frame Prediction for Anomaly Detection, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html

---

## Slide 10. Kiến trúc hệ thống ứng dụng

### Mục đích

Phân biệt hệ thống hoàn chỉnh với một model chạy trong notebook.

### Chữ trên slide

**Luồng xử lý của hệ thống**

1. Nhận video file, webcam hoặc RTSP.
2. Lấy mẫu và tiền xử lý frame.
3. Tạo cửa sổ trượt gồm 8 đến 16 frame.
4. Model tính anomaly score.
5. Làm mượt score theo thời gian.
6. Cảnh báo khi score vượt ngưỡng trong nhiều frame liên tiếp.
7. Hiển thị và lưu nhật ký sự kiện.

**Thông tin lưu khi cảnh báo**

Thời gian, anomaly score, ảnh đại diện và đoạn video ngắn của sự kiện.

### Bố cục và hình ảnh

- Dùng một pipeline ngang hoặc dạng chữ S nếu thiếu chiều rộng.
- Mỗi bước dùng biểu tượng đơn giản, nhãn ngắn và số thứ tự.
- Đường nối không được cắt qua chữ.
- Dùng màu cam duy nhất cho bước cảnh báo.
- Sơ đồ phải có thể chỉnh sửa.

### Ghi chú thuyết trình

Một mô hình đạt AUROC cao chưa tự động trở thành hệ thống ứng dụng. Hệ thống cần xử lý đầu vào camera, duy trì buffer frame, tính điểm liên tục và quyết định khi nào tạo cảnh báo. Nếu chỉ dựa vào một frame vượt ngưỡng, hệ thống dễ báo giả do nhiễu hoặc thay đổi ánh sáng. Vì vậy, dự án dự kiến làm mượt anomaly score và yêu cầu nhiều frame liên tiếp vượt ngưỡng. Demo cuối kỳ cần lưu lại bằng chứng để người vận hành có thể xem lại sự kiện.

---

## Slide 11. Tiêu chí đánh giá

### Mục đích

Trình bày cả metric học thuật và metric phản ánh khả năng sử dụng.

### Chữ trên slide

**Đánh giá trên dataset**

- Frame-level AUROC: khả năng xếp frame bất thường cao hơn frame bình thường.
- Average Precision: phù hợp khi frame bất thường chiếm tỷ lệ nhỏ.
- Precision, Recall và F1 tại một ngưỡng cảnh báo cụ thể.

**Đánh giá hệ thống**

- Số cảnh báo giả trên mỗi giờ video.
- Độ trễ từ khi sự kiện bắt đầu đến khi hệ thống cảnh báo.
- Tốc độ xử lý theo FPS và thời gian inference.

**Nguyên tắc**

Chọn ngưỡng bằng tập validation. Không dùng nhãn test để điều chỉnh ngưỡng hoặc siêu tham số.

### Bố cục và hình ảnh

- Chia slide thành hai phần rõ ràng: Dataset metrics và System metrics.
- Dùng một ROC curve minh họa đơn giản, ghi rõ “Minh họa”.
- Không vẽ biểu đồ với dữ liệu giả giống kết quả thực nghiệm.
- Dòng nguyên tắc về validation đặt ở chân slide và tô màu cam nhạt.

### Ghi chú thuyết trình

AUROC là metric xếp hạng, không phải tỷ lệ dự đoán đúng tại một ngưỡng. Một hệ thống có AUROC tốt vẫn có thể tạo nhiều cảnh báo giả khi triển khai. Vì vậy, dự án cần bổ sung Average Precision, precision, recall và F1. Đối với demo ứng dụng, số cảnh báo giả trên mỗi giờ và độ trễ cảnh báo có ý nghĩa trực tiếp hơn. Ngưỡng phải được xác định từ validation, tốt nhất là một phần dữ liệu bình thường được giữ lại từ tập train.

---

## Slide 12. Thiết kế thí nghiệm

### Mục đích

Xác định các mô hình, biến thể và bảng kết quả sẽ thực hiện trong các tuần sau.

### Chữ trên slide

**Mô hình so sánh**

- Baseline: Conv-Autoencoder.
- Mô hình chính: CNN và Temporal Transformer.
- Mở rộng nếu còn thời gian: ConvLSTM-Autoencoder.

**Ablation ưu tiên**

- Không có Transformer và có Transformer.
- Sequence length 8 và 16.
- Pixel reconstruction và feature prediction nếu đủ thời gian.

**Bảng kết quả dự kiến**

| Model | AUROC | AP | F1 | Báo giả/giờ | FPS |
|---|---:|---:|---:|---:|---:|
| Conv-AE | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |
| CNN-Transformer | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện | Chưa thực hiện |

### Bố cục và hình ảnh

- Phần trên dùng văn bản ngắn cho mô hình và ablation.
- Phần dưới là bảng kết quả có thể chỉnh sửa.
- Không điền kết quả của bài báo vào bảng kết quả của dự án.

### Ghi chú thuyết trình

Thiết kế ban đầu có ba mô hình nhưng phạm vi tám tuần có thể quá lớn đối với người mới học AI. Vì vậy, Conv-Autoencoder và CNN-Transformer là hai mô hình bắt buộc. ConvLSTM chỉ thực hiện khi data pipeline và mô hình chính đã ổn định. Ablation quan trọng nhất là loại bỏ Transformer trong khi giữ các thành phần khác giống nhau. Thí nghiệm này cho biết Transformer có đóng góp thực sự hay không.

---

## Slide 13. Kết quả khảo sát tuần 1 và rủi ro

### Mục đích

Báo cáo đúng những gì đã tìm hiểu và thể hiện nhận thức về các rủi ro chính.

### Chữ trên slide

**Kết quả khảo sát tuần 1**

- Xác định bài toán normal-only video anomaly detection.
- Giới hạn bối cảnh ở camera cố định và khu vực dành cho người đi bộ.
- Khảo sát UCSD Ped2 và CUHK Avenue.
- Đề xuất Conv-Autoencoder làm baseline.
- Thiết kế sơ bộ CNN và Temporal Transformer.
- Xác định pipeline từ video đầu vào đến cảnh báo.

**Rủi ro cần kiểm soát**

| Rủi ro | Hướng xử lý |
|---|---|
| Chưa có nền tảng AI | Học PyTorch theo từng thành phần cần dùng |
| Dataset nhỏ và dễ overfit | Validation riêng, augmentation có kiểm soát |
| Ped2 và Avenue khác miền | Đánh giá cùng miền trước, cross-domain sau |
| Transformer tốn tài nguyên | ResNet18, sequence 8, Transformer 2 lớp |
| Báo giả do ánh sáng hoặc rung | Làm mượt score và kiểm tra nhiều frame liên tiếp |

### Bố cục và hình ảnh

- Chia slide theo tỷ lệ khoảng 40/60.
- Bên trái là kết quả khảo sát.
- Bên phải là bảng rủi ro có thể chỉnh sửa.
- Không thêm ảnh trang trí vì slide đã có nhiều nội dung.

### Ghi chú thuyết trình

Tuần đầu tập trung vào việc làm rõ bài toán trước khi viết model. Rủi ro lớn nhất là mở rộng phạm vi quá nhanh và đánh giá cross-domain như một bài test thông thường. Rủi ro thứ hai là dành quá nhiều thời gian cho Transformer trước khi data pipeline hoạt động. Cách triển khai dự kiến là xây baseline nhỏ trước, kiểm tra toàn bộ quá trình đọc dữ liệu, train, tính score và đánh giá, sau đó mới thay thế phần mô hình bằng CNN-Transformer.

### Chú ý trước khi trình bày

Người dùng phải sửa các gạch đầu dòng trên theo tiến độ thực tế. Nếu chưa thực hiện một nội dung, đổi thành “Đang khảo sát” hoặc chuyển sang kế hoạch tuần tiếp theo.

---

## Slide 14. Kế hoạch thực hiện

### Mục đích

Nêu đầu ra cụ thể của tuần 2 và lộ trình đến cuối dự án.

### Chữ trên slide

**Mục tiêu tuần 2**

- Tải và kiểm tra cấu trúc Ped2, Avenue.
- Thống kê số video, frame, độ phân giải và tỷ lệ frame bất thường.
- Hiển thị frame cùng ground truth để kiểm tra nhãn.
- Viết PyTorch Dataset và DataLoader cho chuỗi 8 frame.
- Tách validation từ dữ liệu bình thường.
- Chạy thử một batch qua CNN nhỏ.

**Lộ trình 8 tuần**

| Tuần | Đầu ra chính |
|---|---|
| 1 | Khảo sát bài toán, dataset và kiến trúc |
| 2 | Data pipeline và thống kê dữ liệu |
| 3 | Conv-Autoencoder baseline |
| 4 đến 5 | CNN-Transformer và huấn luyện |
| 6 | Đánh giá, ablation và cross-domain test |
| 7 | Demo video, cảnh báo và nhật ký |
| 8 | Báo cáo, kiểm tra và chuẩn bị bảo vệ |

### Bố cục và hình ảnh

- Nửa trái là mục tiêu tuần 2.
- Nửa phải là timeline 8 tuần có thể chỉnh sửa.
- Nhấn tuần 2 bằng màu teal.
- Không dùng timeline quá nhiều chi tiết hoặc chữ nhỏ.

### Ghi chú thuyết trình

Đầu ra quan trọng nhất của tuần 2 là data pipeline đúng. Cần kiểm tra trực quan ground truth, vì chỉ một lỗi lệch frame cũng làm toàn bộ metric sai. Mỗi mẫu đầu vào dự kiến có dạng batch, sequence, channel, height và width. Các sequence không được vượt qua ranh giới giữa hai video. Sau khi DataLoader ổn định, tuần 3 mới bắt đầu huấn luyện Conv-Autoencoder.

---

## Slide 15. Kết luận tuần 1

### Mục đích

Tóm tắt các quyết định chính và điều kiện để dự án thành công.

### Chữ trên slide

**Kết luận tuần 1**

- Đề tài phù hợp với học phần và khả thi trong 8 tuần nếu giữ phạm vi camera cố định.
- Ped2 phù hợp để phát triển mô hình. Avenue phù hợp để xác nhận phương pháp và khảo sát cross-domain.
- Conv-Autoencoder là baseline bắt buộc. CNN-Transformer là mô hình chính.
- Kết quả cần được đánh giá bằng cả metric học thuật và số cảnh báo giả.
- Giá trị ứng dụng đến từ toàn bộ pipeline cảnh báo, không chỉ từ model.

**Bước tiếp theo**

Hoàn thiện data pipeline và kiểm tra dữ liệu trước khi huấn luyện.

### Bố cục và hình ảnh

- Slide kết luận đơn giản, nhiều khoảng trống.
- Có thể dùng một ảnh CCTV nhỏ ở góc phải hoặc một dải frame mờ ở chân slide.
- Làm nổi bật dòng “Bước tiếp theo”.
- Không thêm slide “Cảm ơn” riêng nếu thời lượng bị giới hạn.

### Ghi chú thuyết trình

Kết quả khảo sát cho thấy đề tài có thể thực hiện trong thời gian môn học nếu ưu tiên đúng thứ tự. Dự án sẽ bắt đầu bằng một baseline đơn giản, sau đó mới phát triển CNN-Transformer. Hai dataset được sử dụng theo split chính thức trước khi thực hiện thí nghiệm cross-domain. Mục tiêu cuối cùng không chỉ là đạt một chỉ số AUROC, mà còn tạo được demo tiếp nhận video, tính điểm bất thường, giảm cảnh báo giả và lưu lại sự kiện để người vận hành xem xét.

---

## 3. Tài liệu tham khảo cho speaker notes

Plugin cần gắn các nguồn sau vào speaker notes của slide liên quan. Không cần hiển thị toàn bộ URL trên từng slide nếu bố cục chật.

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

## 4. Câu hỏi và câu trả lời chuẩn bị cho phần trao đổi

### Vì sao chọn cách học chỉ từ video bình thường?

Sự kiện bất thường hiếm và không thể thu thập đầy đủ mọi loại. Học từ hoạt động bình thường giúp model có cơ hội phát hiện cả những mẫu bất thường chưa xuất hiện trong dữ liệu huấn luyện.

### Vì sao chọn Ped2?

Ped2 có dung lượng nhỏ, split rõ ràng và ground truth theo frame. Dataset phù hợp để người mới xây dựng và kiểm tra toàn bộ pipeline trong thời gian giới hạn.

### Vì sao cần Avenue?

Avenue đa dạng hơn Ped2 và có thông tin vùng bất thường. Dataset giúp kiểm tra liệu cùng phương pháp có hoạt động trên một bối cảnh khác hay không.

### Có thể train Ped2 rồi test Avenue không?

Có thể, nhưng đây là cross-domain evaluation. Sự khác biệt về camera, nền và màu sắc có thể làm anomaly score tăng ngay cả khi hành vi bình thường. Vì vậy, kết quả cần được trình bày như một thí nghiệm tổng quát hóa, không thay thế đánh giá theo split chính thức.

### Vì sao dùng Transformer?

CNN lấy đặc trưng trong từng frame. Transformer phân tích quan hệ giữa nhiều frame liên tiếp, từ đó hỗ trợ phát hiện các chuyển động không theo quy luật đã học.

### Làm sao biết Transformer thực sự có ích?

So sánh hai cấu hình giống nhau, chỉ thay đổi việc có hoặc không có Transformer. Nếu các điều kiện huấn luyện giống nhau, chênh lệch kết quả thể hiện đóng góp của khối temporal attention.

### Hệ thống có chạy thời gian thực không?

Dự án chưa cam kết trước khi đo. ResNet18 và Transformer nhỏ được chọn để giảm chi phí tính toán. Giai đoạn đánh giá sẽ báo cáo FPS và latency trên thiết bị thực tế.

### Ngưỡng cảnh báo được chọn như thế nào?

Giữ lại một phần dữ liệu bình thường làm validation. Phân bố anomaly score của validation được dùng để đặt ngưỡng. Không dùng nhãn test để tìm ngưỡng vì sẽ gây data leakage.

### AUC cao có nghĩa hệ thống ứng dụng tốt không?

Không hoàn toàn. AUROC cho biết khả năng xếp hạng frame nhưng không phản ánh trực tiếp số cảnh báo giả ở một ngưỡng cụ thể. Dự án cần báo cáo thêm precision, recall, F1, cảnh báo giả trên mỗi giờ và độ trễ.

### Hệ thống có phát hiện được đánh nhau hoặc trộm cắp không?

Không nên khẳng định với Ped2. Dataset này chủ yếu phù hợp với đối tượng và chuyển động bất thường trong lối đi bộ. Muốn xử lý bạo lực hoặc trộm cắp cần dataset và cách giám sát khác.

---

## 5. Kiểm tra trước khi giao plugin presentation

- Đã thay toàn bộ placeholder thông tin cá nhân.
- Đã cập nhật tiến độ thực tế ở slide 13.
- Không có số liệu huấn luyện chưa được đo.
- Các hình Ped2 và Avenue ghi rõ là hình từ dataset.
- Sơ đồ và bảng vẫn có thể chỉnh sửa.
- Không có đoạn văn dài trong phần hiển thị của slide.
- Speaker notes chứa phần giải thích chi tiết và nguồn.
- Font hiển thị đúng dấu tiếng Việt.
- Slide 7 phân biệt đánh giá cùng miền và cross-domain.
- Slide 11 phân biệt AUROC với accuracy.
- Slide cuối nêu rõ bước tiếp theo là hoàn thiện data pipeline.
