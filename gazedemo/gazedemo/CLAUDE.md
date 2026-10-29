# gazedemo — hướng dẫn cho Claude Code

## Đọc trước, theo thứ tự
1. `docs/CONTEXT.md` — dự án là gì, module này phục vụ gì.
2. `docs/SPEC.md` — phạm vi và yêu cầu đợt này. **Nguồn quyết định.**
3. `docs/SPEC_full.md` — thiết kế dài hạn. Chỉ để hiểu lý do thiết kế và không chặn đường mở rộng; không implement phần nằm ngoài SPEC.md.

## Môi trường
- Máy Windows ở quán net, GPU NVIDIA, webcam USB, thời gian có hạn, máy có thể bị reset khi khởi động lại.
- Code chạy trên Windows: dùng `pathlib`, không lệnh shell Unix trong code, mở camera bằng DirectShow (config `source.backend`).
- Lệnh gợi ý cho người dùng viết theo PowerShell.

## Cách làm
- Làm theo bảng mốc M0 → M7 (SPEC mục 10). Cuối mỗi mốc: chạy `python -m pytest -m "not gpu and not camera" -q`, báo cáo ngắn (đã làm gì, lệnh để thử, người dùng cần nhìn thấy gì), rồi DỪNG chờ xác nhận.
- Ưu tiên khi thiếu giờ: tới M5 trước (thu được data), M6–M7 làm sau cũng được.
- Script Gazelle cũ của người dùng: `~/Documents/FLAIR/bite_acquisition/scripts/gazelle_bridge/server.py` (máy Linux của người dùng). Cách load (đã dùng lại trong `perception/gazelle_model.py`): mã nguồn local + `get_gazelle_model(name)` + `load_gazelle_state_dict(torch.load(ckpt, weights_only=True))`. Mã nguồn + checkpoint cho dự án này: `../../gazelle-main (1)/gazelle-main` (config `gazelle.repo`).
- Sau mỗi mốc, nhắc người dùng sao lưu (`git commit` + push, hoặc copy thư mục ra USB/Drive).

## Ràng buộc
- Không đổi định nghĩa dwell/metric/p_look (SPEC 1, 4.3–4.5) khi chưa hỏi.
- `gazedemo/core/` chỉ import numpy và `gazedemo.types`.
- Mọi ngưỡng trong `configs/default.yaml`.
- Timestamp = `time.monotonic()` lúc chụp frame.
- Gazelle chạy local, `bboxes = [[None]]`.
- Thư viện cho phép: torch, torchvision, timm, opencv-python, numpy, pyyaml, pytest; tuỳ chọn scipy, matplotlib (code phải chạy khi thiếu chúng). Thêm gì khác phải hỏi.
- p_look (xác suất nhìn) và p_choice (xác suất chọn, từ model) luôn là hai tên riêng trên UI và trong file.
- Không làm: ArUco, MediaPipe/head box, ROS, ghép feedbot.
