# Chạy gazedemo ở quán net — checklist

Thời gian ước tính: Claude Code viết code M0–M5 mất vài giờ; tổng thời gian thực tế chưa biết. Ưu tiên **thu được data trước khi về**.

## A. Mang theo
- [ ] Webcam USB (quán net có thể không có) + dây nối dài nếu cần — hoặc D435i + cáp USB 3 (cổng USB 2 chỉ được 15 fps)
- [ ] Thư mục `gazelle-main` có `checkpoints/` (ViT-L, ViT-B) đặt cạnh thư mục `gazedemo` như trên máy nhà (không có thì `torch.hub` tải qua mạng, DINOv2 ViT-L ~1.2 GB)
- [ ] Giá đỡ webcam (tripod mini / kẹp), băng dính
- [ ] 3 đĩa (đĩa giấy cũng được), dán nhãn A, B, C
- [ ] 6 thẻ món in ảnh/tên to (khớp `configs/foods.yaml`)
- [ ] USB để sao lưu (phòng khi không đẩy được lên mạng)
- [ ] Script Gazelle cũ (GitHub / Drive / USB)
- [ ] Thông tin đăng nhập: Claude, GitHub/Google Drive

## B. Ở quán, trước khi gọi Claude Code
1. Chọn máy có GPU NVIDIA. Mở PowerShell, chạy `nvidia-smi` → phải thấy tên GPU.
2. Hỏi quán: có cho cài phần mềm, có reset ổ cứng khi tắt máy không. **Không khởi động lại máy giữa chừng.**
3. Cài Python 3.11 từ python.org (chọn "Install for current user", tick "Add python.exe to PATH").
4. Cài Git for Windows và Claude Code theo hướng dẫn chính thức (docs.claude.com), đăng nhập.
5. Tải `gazedemo.zip`, giải nén vào nơi dễ tìm, ví dụ `D:\gazedemo` (tránh ổ C nếu quán reset ổ C).
6. Môi trường Python (PowerShell, trong thư mục gazedemo):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # nếu bị chặn: Set-ExecutionPolicy -Scope Process Bypass
pip install --upgrade pip
# torch bản CUDA: lấy lệnh đúng ở pytorch.org (Windows, pip, CUDA); ví dụ:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python numpy pyyaml pytest scipy matplotlib
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m gazedemo doctor      # kiểm CUDA, camera, Gazelle (đừng gọi torch.hub.load trực tiếp: gazelle/utils.py import scikit-learn, gazedemo tự xử lý)
```
   Nếu `gazelle-main` không nằm ở `../../gazelle-main (1)/gazelle-main` so với thư mục dự án: thêm `--set "gazelle.repo=D:/duong/dan/gazelle-main"` hoặc sửa `configs/default.yaml`.

7. `git init` trong thư mục và tạo repo GitHub riêng tư để push sau mỗi mốc (hoặc dùng USB). `video.mp4` đã bị `.gitignore` loại (quá lớn) → video chỉ sao lưu bằng USB/Drive.
8. (Đã điền) đường dẫn script Gazelle cũ trong `CLAUDE.md`.

## C. Kiểm từng mốc trên máy quán
Code M0–M7 đã viết xong và test ở nhà (`python -m pytest -m "not gpu and not camera" -q` → toàn bộ pass). Ở quán chỉ cần chạy và nhìn; lỗi gì thì mở Claude Code trong thư mục `gazedemo` và dán output.

Kiểm sau mỗi mốc (chỉ cần nhìn, không cần đọc code):

| Sau | Bạn làm | Đúng khi |
| --- | --- | --- |
| M0 | `python -m gazedemo doctor`, `python -m gazedemo bench` | CUDA True, thấy camera, có số Hz |
| M1 | Xem Claude Code báo test pass | Có test ví dụ A 500/1/200, C 250/1/750 |
| M2 | `python -m gazedemo setup-aoi` | Elip ôm đúng 3 đĩa |
| M3 | `python -m gazedemo live`, nhìn A 3 s rồi B 2 s, S/E | Bảng: A DD ≈ 3000, B ≈ 2000 |
| M4 | `python -m gazedemo cue` | In accuracy + confusion |
| M5 | `python -m gazedemo collect --participant P01 --trials 5` | 5 lượt chạy trọn, trials.csv có chosen_food |

Nếu Hz thấp (< ~5): đổi `gazelle.name` sang `gazelle_dinov2_vitb14_inout` trong config, chạy lại bench.
`doctor` báo camera KHÔNG: dùng index nó gợi ý (`--set source.device=N`); webcam thường giữ `source.fourcc: MJPG`, D435i cần `source.fourcc: null`.

## D. Thu dữ liệu (sau khi M5 xong)
1. `setup-aoi` lại nếu đã xê dịch đĩa/camera.
2. `cue`: 3 đĩa × 5 = 15 lượt (~2 phút) → con số "nhìn đĩa nào".
3. `collect --participant P01 --trials 30` (~10–15 phút). Chọn món thật lòng muốn, nhìn tự nhiên.
4. Còn thời gian: thêm một buổi `collect` nữa (nghỉ 5 phút, đặt lại đĩa) → 60 lượt.
5. Quay màn hình 1–2 phút chạy `live` để show (Win + Alt + R nếu Xbox Game Bar có; không thì quay bằng điện thoại).

## E. Trước khi về — BẮT BUỘC
- [ ] `git add -A; git commit -m "data + code"; git push` (video.mp4 bị `.gitignore` loại vì quá lớn → copy `runs/` ra USB/Drive để giữ video cho `offline`).
- [ ] Copy cả thư mục `gazedemo` (gồm `runs/`) ra USB.
- [ ] Nếu M6 chưa xong: không sao, `dataset` + `train` chạy được ở nhà không cần GPU/camera.
- [ ] Đăng xuất Claude, GitHub, Google trên máy quán.
