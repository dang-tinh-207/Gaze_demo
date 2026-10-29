# Báo cáo nhanh: module gaze → TTFD/DD/DC/p_look → classifier chọn món

Điền số từ `bench.md`, `score.md`, `models/report.md` (hoặc `report/SUMMARY.md`). Ô chưa có số để "?".

## 1. Module làm gì
Camera ngoài → Gazelle (ViT-? , ? Hz trên GPU ?) → mỗi frame gán đĩa A/B/C/none theo phần heatmap rơi vào elip đĩa → gom thành dwell → mỗi lượt xuất cho từng đĩa:

| Feature | Ý nghĩa |
| --- | --- |
| TTFD | Thời gian từ đầu lượt tới lần nhìn đầu tiên vào đĩa |
| DD | Tổng thời gian nhìn đĩa |
| DC | Số lần ghé đĩa |
| p_look | Xác suất đang nhìn đĩa (trung bình heatmap trong đĩa, chuẩn hoá) |
| p_choice | Xác suất chọn món, từ classifier đã train |

Ví dụ một lượt thật (copy 1 dòng từ `trials.csv`): ?

## 2. "Nhìn đĩa nào" có đúng không (cue, 1 người, 15 lượt)
- Accuracy (đĩa có DD lớn nhất = đĩa được gọi): ? / 15
- Confusion matrix: ?
- Tỉ lệ thời gian hợp lệ trung bình: ?

## 3. Dữ liệu chọn món đã thu
- Số lượt: ? (1 người), mỗi lượt ? s, 3 món / 6 món, vị trí món cân bằng giữa các đĩa.
- Ground truth: đĩa được chọn, nhập sau tiếng bíp; chỉ dùng gaze trước bíp.

## 4. Classifier (conditional logit, K-fold theo lượt)

| Mô hình | Top-1 | Log-loss | CI 95% |
| --- | --- | --- | --- |
| Chance | 0.33 | 1.10 | — |
| Argmax DD | ? | ? | ? |
| Đĩa nhìn cuối | ? | ? | ? |
| Chỉ mức thích | ? | ? | ? |
| Logit G1 (TTFD/DD/DC) | ? | ? | ? |
| Logit all | ? | ? | ? |
| Logit all + mức thích | ? | ? | ? |

## 5. Giới hạn (nói thẳng)
- 1 người, cũng là người thiết kế thí nghiệm → biết mục đích, có thể nhìn "có chủ đích". Kết quả chứng minh pipeline chạy, chưa phải bằng chứng tổng quát.
- AOI vẽ tay, chưa có head box, chưa có ArUco.
- Gazelle chưa được đánh giá trên rig ăn uống (dữ liệu train là cảnh chung).

## 6. Bước tiếp
- Thu nhiều người (≥ ? người × 40 lượt) → LOSO.
- ArUco cho AOI tự động; so head box none vs fixed.
- Ghép vào framework: p_choice → fusion với speech và prior.
