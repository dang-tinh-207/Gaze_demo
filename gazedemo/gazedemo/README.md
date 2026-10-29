# gazedemo

Người dùng đang nhìn đĩa nào (A/B/C) bằng Gazelle chạy local → mỗi lượt xuất TTFD (TFD), DD, DC, p_look theo từng đĩa/món → thu ground truth → train classifier ra p_choice.

## File
| File | Cho ai | Nội dung |
| --- | --- | --- |
| `docs/RUNBOOK_quan_net.md` | Bạn | Mang gì, cài gì, kiểm gì sau mỗi mốc, sao lưu trước khi về |
| `CLAUDE.md` | Claude Code | Thứ tự đọc, môi trường Windows, ràng buộc |
| `docs/CONTEXT.md` | Claude Code | Bối cảnh dự án |
| `docs/SPEC.md` | Claude Code | Đặc tả đợt này (nguồn quyết định), mốc M0–M7 |
| `docs/SPEC_full.md` | Claude Code | Thiết kế dài hạn |
| `docs/DEMO_SUPERVISOR.md` | Bạn | Khung báo cáo; `report` tự điền thành `report/SUMMARY.md` |
| `configs/default.yaml`, `configs/foods.yaml` | Cả hai | Ngưỡng, camera, danh sách món |

## Bắt đầu
Làm theo `docs/RUNBOOK_quan_net.md` mục B rồi C. Gazelle được load từ mã nguồn local `gazelle.repo` (mặc định `../../gazelle-main (1)/gazelle-main`, checkpoint trong `checkpoints/`); không thấy thư mục đó thì tải qua `torch.hub` (cần mạng).

## Lệnh
```powershell
python -m gazedemo doctor                      # Python/torch/CUDA/GPU/camera/Gazelle
python -m gazedemo bench --n 200               # → runs/<id>_bench/bench.md
python -m gazedemo setup-aoi                   # vẽ 3 elip → aoi.json + aoi_ref.png
python -m gazedemo live                        # S/E trial, 1/2/3 nhãn, H heatmap, Q thoát
python -m gazedemo cue                         # nhìn theo hiệu lệnh → in score
python -m gazedemo score --run runs/<id>       # accuracy, confusion, reliability/ECE → score.md
python -m gazedemo calibrate --run runs/<cue>  # → models/calibration.json (thêm p_look_cal)
python -m gazedemo collect --participant P01 --trials 30
python -m gazedemo offline --run runs/<id>     # chạy lại trên video.mp4 → runs/<id>/offline/
python -m gazedemo dataset --runs runs --out data/gaze_choice.csv
python -m gazedemo train --data data/gaze_choice.csv --out models
python -m gazedemo live --model models/model.json   # thêm cột p_choice
python -m gazedemo report --out report         # → report/SUMMARY.md (+ overlay.png, timeline.png)
python -m gazedemo synth --participants 1 --trials 60 --out runs_synth   # dữ liệu giả để thử dataset/train
```

- Ghi đè config mà không sửa file: `--set khoa=giatri` (đặt trước hoặc sau tên lệnh), vd `--set source.device=1`, `--set gazelle.name=gazelle_dinov2_vitb14_inout`.
- Thử không cần camera/GPU: `python -m gazedemo live --source synthetic --fake-gaze`. `--source` nhận số camera, `synthetic`, file ảnh hoặc video.
- Mỗi lệnh có camera ghi `runs/<YYYYmmdd_HHMMSS>_<lệnh>/`: `video.mp4`, `frames.csv`, `samples.jsonl`, `events.jsonl`, `trials.csv`, `dwells.csv`, `config.yaml`, `aoi.json`, `run.json` (+ `session.json`, `schedule.csv` với collect; `score.md` với cue).
- `video.mp4` ≈ 70 MB/phút (720p, 15 fps) → không đẩy lên git (đã có `.gitignore`), sao lưu `runs/` bằng USB/Drive.

## Camera
- Windows: DirectShow, webcam thường dùng `source.fourcc: MJPG` để đạt 1280×720 @ 30 fps.
- Intel RealSense D435i (chỉ dùng luồng màu): `source.fourcc: null`; mỗi camera RealSense hiện nhiều index (màu / hồng ngoại / depth) → `doctor --max-index 12` liệt kê index mở được, chọn index cho ảnh màu 1280×720. Cắm cổng USB 3 (USB 2 chỉ được 15 fps). Không chạy ROS `realsense2_camera` cùng lúc.
- Linux có `librealsense2-dkms`: driver `uvcvideo` được build lại và ký bằng khoá riêng; khi Secure Boot bật phải enroll khoá (`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`, khởi động lại, Enroll MOK) thì mới có `/dev/video*`.

## Test
```powershell
python -m pytest -m "not gpu and not camera" -q   # mỗi mốc (không cần GPU/camera)
python -m pytest -q                               # thêm test GPU (Gazelle thật) + camera
``` 
