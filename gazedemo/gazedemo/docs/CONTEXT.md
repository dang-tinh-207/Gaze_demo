# Bối cảnh dự án (để hiểu module này phục vụ gì)

## Dự án
Robot hỗ trợ ăn (robot-assisted feeding), đưa thức ăn ra trước miệng (outside-mouth transfer). Team: một người làm phần cứng (gripper tự đổi dụng cụ theo món), người dùng repo này làm phần software. Supervisor: anh Tuấn.

Đóng góp software: framework cá nhân hoá — tự điều chỉnh việc chọn món, chọn dụng cụ và kênh tương tác theo trạng thái người dùng trong bữa (mệt, kênh hỏng), khởi tạo từ hồ sơ chức năng của nhóm bệnh. Các hệ hiện có (FEAST, FLAIR, …) bắt người dùng tự cấu hình lại khi trạng thái đổi.

## Chế độ theo độ tin cậy kênh
- Mode 1: gaze (hướng đầu) + speech, tốc độ thường.
- Mode 2: speech only, chậm.
- Mode 3: gaze only, chậm.
- Mode 4 (trạng thái tạm): không kênh nào → thói quen ăn + lịch sử gắp.
Kênh hỏng → "chuyển kênh tạm thời", xác nhận qua kênh khác khi độ tin cậy thấp.

## Module gaze nằm ở đâu
Chọn món tiếp theo = P(món) ∝ softmax(w_g·log p_gaze + w_p·log p_prior):
- ≥ τ_cao: tự chọn; giữa τ_thấp và τ_cao: hỏi xác nhận bằng speech; dưới τ_thấp: dùng prior.
- Speech rõ luôn thắng; "dừng" + nút cứng ngắt mọi bước.
Module này cung cấp p_gaze: đĩa nào được nhìn bao lâu, bao nhiêu lần, nhìn đến sớm hay muộn (DD/DC/TTFD + `mass`). Classifier (conditional logit) sau này học từ các feature đó.

## Tách hai câu hỏi
1. "Nhìn đĩa nào" — module này, đo được bằng nhìn theo hiệu lệnh.
2. "Muốn ăn món nào" — classifier, cần dữ liệu chọn món thật (protocol 2a: ~40 lượt/người, bíp cuối cửa sổ, chỉ dùng gaze trước bíp, đảo vị trí món giữa A/B/C).

## Bằng chứng tham khảo (định hướng feature, chưa phải kết quả của mình)
- Danner et al. 2016 (Acta Alimentaria, 59 người, 4 món): món được chọn có tổng dwell dài hơn, dwell count cao hơn; first fixation/TTFF không tương quan với lựa chọn.
- Gaze cascade (Shimojo et al. 2003): nhìn dồn về lựa chọn cuối ngay trước khi quyết định → feature "DD trong 1–2 s cuối", "đĩa nhìn cuối".
- Stiefelhagen & Zhu 2002: hướng đầu đóng góp phần lớn hướng nhìn → lý do chấp nhận gaze ước lượng từ đầu/ảnh cả cảnh.

## Trạng thái (28/10/2026)
- Gazelle chạy trên ảnh tĩnh với hộp đĩa + hộp đầu cho kết quả "tương đối ổn"; chưa có số đo.
- Chưa thu dữ liệu chọn món.
- Đợt này (chạy ở quán net có GPU, 1 buổi): demo live TTFD/DD/DC/p_look; accuracy "nhìn đĩa nào" bằng cue; thu ~30–60 lượt chọn món của chính người dùng; train/eval classifier để show supervisor rằng pipeline feature ↔ ground truth → model chạy trọn.
- Repo tổng (perception gồm food detection, nhiều model giao tiếp qua ROS) là việc cả team, làm sau. Module này phải dễ gói thành ROS node: giữ core thuần, output = một bản ghi per-trial (DD/DC/TTFD theo đĩa + timestamp).
