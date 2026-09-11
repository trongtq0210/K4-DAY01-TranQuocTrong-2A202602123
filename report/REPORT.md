# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915 
    đây là record hạng 1 vì có score cao nhất trong số sample_id=trafic

- Record này mô tả toàn ảnh như thế nào?
    Đây là một prediction cấp ảnh: mô hình cho rằng ảnh `traffic` có xác suất cao nhất thuộc lớp `cab` trên toàn bộ khung hình. Nó mô tả sự gán nhãn cho cả ảnh chứ không phải cho từng đối tượng riêng lẻ.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    Người định nghĩa danh sách các lớp (class list) chính là con người, cụ thể là những kỹ sư hoặc chuyên gia tạo ra bộ dữ liệu gốc từ ban đầu. Quá trình này được quyết định từ trước khi mô hình được mang đi huấn luyện. Do đó, mô hình (checkpoint) chỉ có thể học và đưa ra dự đoán dựa trên đúng danh sách mà con người đã cung cấp sẵn, chứ máy tính không có khả năng tự phát minh hay tự tạo ra tên gọi mới cho những vật thể nằm ngoài danh sách đó.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    Cần giữ cả ba để máy và người đều hiểu đúng. ID giúp máy tính xử lý nhanh. Tên lớp để con người dễ đọc hiểu. Còn taxonomy giống như phần ghi chú nguồn gốc, giúp tránh nhầm lẫn từ vựng giữa các bộ dữ liệu khác nhau (ví dụ: để biết là "chuột" máy tính hay con "chuột").

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Guideline cần quy định rõ luật ưu tiên để giải quyết xung đột (ví dụ: ưu tiên vật chiếm diện tích lớn nhất, vật ở trung tâm, hoặc chủ thể rõ nét nhất). Nếu không có luật này, dữ liệu sinh ra sẽ bị nhiễu do mỗi người làm một kiểu.
    Ví dụ: Một bức ảnh đường phố có cả "người đi bộ" và "ô tô", guideline phải chốt rõ: nếu bài toán đang làm là camera giao thông phạt nguội thì bắt buộc ưu tiên chọn nhãn "ô tô", nếu là hệ thống AI cảnh báo va chạm thì ưu tiên chọn "người đi bộ".

- Vì sao model score không phải ground truth?
    score chỉ là xác suất tính toán thống kê do thuật toán của máy xuất ra; ground truth là nhãn chuẩn thực tế do con người thẩm định. Mô hình hoàn toàn có thể bị nhầm lẫn và cho điểm tự tin cực cao vào một kết quả sai.
    Ví dụ: Mô hình YOLO nhìn nhầm một chiếc túi nilon màu đen bay trên đường và dự đoán đó là "con chim" với score lên tới 0.95. Dù điểm máy chấm có cao đến đâu, ground truth thực tế do con người xác nhận vẫn phải là "túi nilon" hoặc background.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    "sample_id": "kitchen",
    "class_name": "person",
    "score": 0.899318,
    "bbox_xyxy": [0.08, 256.79, 18.39, 313.12],
    "bbox_width": 18.32,
    "bbox_height": 56.33
    Đây là một prediction của object detection cho một đối tượng trong ảnh `kitchen`. Có class_name là `person`, score 0.899318, và box được ghi bằng tọa độ pixel `xyxy`.

- Diễn giải vị trí box bằng lời:
    Box này bắt đầu ở x = 0.08, y = 256.79 và kết thúc ở x = 18.39, y = 313.12. Nói cách khác, đối tượng nằm ở mép trái của ảnh, có bề ngang khoảng 18.32 pixel và bề cao khoảng 56.33 pixel. Tức là đây là một người ở góc ảnh, không phải một đối tượng lớn như bàn hay tủ.

- So sánh số prediction ở hai threshold:
    Theo notebook, với `sample_id = "kitchen"`, khi dùng threshold 0.20 thì số prediction là 11, khi threshold 0.35 cũng là 11, nhưng khi lên đến 0.60 thì chỉ còn 6. Điều này cho thấy nếu tăng ngưỡng, mô hình loại bỏ nhiều box yếu, còn nếu giảm ngưỡng thì model sẽ phát hiện nhiều object hơn nhưng cũng dễ mắc lỗi false positive.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    Nếu threshold thấp, độ bao phủ lớn hơn, nghĩa là nhiều vật thể được phát hiện hơn, nhưng reviewer phải xem nhiều hơn và làm thêm công việc lọc false positive. Nếu threshold cao, workload giảm nhưng có nguy cơ bỏ sót object thật vì box có điểm thấp hơn sẽ bị loại.

- Đề xuất một quy tắc box chặt:
    Quy tắc box chặt nên là “bọc sát phần hình ảnh thực sự thuộc vật thể”, không kéo dài quá nền và không mở rộng quá xa ngoài biên đối tượng. Box phải vừa đủ để bao cả vật thể mà không lẫn nhiều đồ vật lân cận.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    guideline phải quy định rõ cách xử lý object bị che một phần hoặc bị cắt ở mép ảnh. Nếu không chắc có nên giữ box hay bỏ box thì annotator cần escalate cho reviewer để cùng quyết định, không được tự ý kéo box quá rộng hay thêm phần không nhìn thấy.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    "sample_id": "kitchen",
    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_point_count": 348,
    "polygon_xy": [[...], [...], ...]
    Đây là một record instance segmentation, nghĩa là mô hình không chỉ cho biết class và box mà còn cho biết hình dạng thực tế của đối tượng bằng polygon.

- Polygon bổ sung chi tiết gì so với box?
    Box chỉ là một hình chữ nhật bao quanh vật thể. Polygon thì cho biết đúng biên dạng của đối tượng, ví dụ các góc, đường cong, mảng cắt uốn của người trong ảnh. Vì vậy polygon có độ chính xác cao hơn box và rất hữu ích cho việc training hay QC mặt nạ.

- `instance_id` dùng để làm gì và không phải loại ID nào?
    `instance_id` dùng để phân biệt từng instance khác nhau trong cùng một ảnh. Ví dụ nếu có hai người cùng class `person`, thì phải có hai `instance_id` riêng để kiểm soát từng đối tượng. Nó không phải class_id, vì class_id là mã lớp chung, không phải mã của từng instance.

- Đề xuất một quy tắc biên mask:
    Biên mask nên khép kín, sát với contour thật của vật thể, không bao gồm nền và không chồng lên vật thể khác. Nếu một phần không nhìn thấy rõ thì nên giữ phần nhìn thấy được, nhưng không “đoán” quá mức lên phần bị che.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Cần có guideline rõ ràng cho vùng mờ, vùng chồng lẫn, và vùng bị che. Nếu annotator không chắc chắn về biên của mask thì phải gửi lên reviewer để quyết định, vì mask sai có thể làm lệch dữ liệu huấn luyện và QC rất khó phát hiện.


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Mỗi ảnh gán 1 label chính theo taxonomy (ví dụ `cab`, `restaurant`) | Một ảnh có nhiều đối tượng, label có thể mơ hồ nếu không có quy tắc ưu tiên | Chọn class chính theo guideline; nếu cần, ghi rõ các trường hợp mơ hồ và tham khảo QC | Xem liệu label có đúng taxonomy, đúng class chính và phù hợp với guideline không |
| Phát hiện vật thể | Một hoặc nhiều box cho từng vật thể, mỗi box có `class_name`, `score`, `bbox_xyxy` | Box quá lớn, quá nhỏ, hoặc bao cả nền; vật thể bị cắt mép/che khuất | Tạo box sát vật thể thật nhìn thấy, dùng quy tắc “tight box”; đánh dấu occluded/truncated nếu cần | Kiểm tra box có kín, không chồng chéo sai, có phù hợp với object không |
| Instance segmentation | Mỗi instance có `instance_id`, mask polygon và box; đối tượng cùng lớp vẫn tách riêng | Biên mask quá rộng, nhòe, chồng lẫn giữa các instance, vùng che khuất không rõ | Vẽ polygon theo contour nhìn thấy và tách riêng từng instance | Kiểm tra tính đầy đủ, không chồng lẫn và biên mask phù hợp với guideline |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Chỉ sử dụng ảnh công khai theo giấy phép và không lưu trữ dữ liệu cá nhân, thông tin nhạy cảm hoặc dữ liệu nội bộ ngoài phạm vi lab.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Lab Coach / Giảng viên phụ trách hoặc người quản lý bài lab để xác minh và dừng xử lý ngay.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

### Kết luận ngắn

Bài lab cho thấy prediction của mô hình chỉ là đầu ra của model theo taxonomy đã huấn luyện; nó không thay thế ground truth do con người tạo theo guideline. Điểm cốt lõi là đọc đúng đơn vị nhãn của từng tác vụ: ảnh-level label cho classification, object-level box cho detection, và instance-level polygon cho segmentation. Việc kiểm tra chất lượng cần dựa trên nguyên tắc tight box, closed mask, rõ quy tắc occlusion/truncation và review thống nhất để đảm bảo giá trị annotation.