# gazedemo — đặc tả đợt này (NGUỒN QUYẾT ĐỊNH)

Phiên bản 28/10/2026. Khi file này khác `SPEC_full.md`, **file này thắng**.

Mục tiêu đợt này, cho buổi gặp supervisor:
1. Chạy live: cho mỗi đĩa A/B/C hiện **TTFD (= TFD), DD, DC và xác suất nhìn** (p_look), kèm tên món đang ở đĩa đó.
2. Đo độ đúng "nhìn đĩa nào" bằng nhìn theo hiệu lệnh → accuracy + confusion matrix.
3. Thu dữ liệu chọn món: feature (TTFD, DD, DC, p_look, …) ↔ ground truth (đĩa/món được chọn).
4. Xuất dataset và train/eval classifier → bảng so sánh với baseline; live hiện thêm **xác suất chọn** (p_choice) nếu đã có model.

Ngoài phạm vi: ArUco, MediaPipe/head box, ROS, ghép repo feedbot, LOSO nhiều người.

Môi trường chạy: **máy Windows ở quán net có GPU NVIDIA**, webcam USB, phiên làm việc có giới hạn thời gian, ổ cứng có thể bị reset khi khởi động lại. Code phải chạy được trên Windows (đường dẫn dùng `pathlib`, không dùng lệnh shell Unix trong code).

---

## 1. Định nghĩa

| Tên | Ký hiệu | Định nghĩa | Đơn vị |
| --- | --- | --- | --- |
| Time to first dwell | TTFD (alias TFD) | Từ t0 đến lúc bắt đầu dwell đầu tiên trên đĩa; rỗng nếu không ghé; đang nhìn sẵn lúc t0 → 0 | ms |
| Dwell duration | DD | Tổng thời lượng các dwell trên đĩa trong [t0, t1) | ms |
| Dwell fixation | DC | Số dwell (số lần ghé) trên đĩa trong [t0, t1) | lần |
| Xác suất nhìn | p_look | Mục 1.1. Mỗi đĩa + "none"; tổng = 1 | 0–1 |
| Xác suất chọn | p_choice | Output của classifier đã train (mục 7); chỉ có khi có `model.json` | 0–1 |

- **Dwell** = lần nhìn liên tục vào một đĩa, dài ≥ `min_dwell_ms` (mục 4.3).
- "Dwell fixation" là số lần ghé, không phải fixation của eye tracker (Gazelle không đo được saccade).
- p_look trả lời "đang nhìn đĩa nào", **không** phải "muốn ăn món nào". p_choice mới là xác suất chọn món. Trên UI và trong file phải ghi rõ hai tên khác nhau.
### 1.1 p_look được tính thế nào
Heatmap Gazelle 64×64, mỗi ô một giá trị trong [0, 1] ("ô này có khả năng là điểm nhìn đến đâu"). Các ô **không** cộng lại bằng 1, nên không phải phân phối xác suất; phải tự dựng.

Mỗi frame:
1. `P = max(H, 0)^gamma`, chuẩn hoá `P / ΣP` → phân phối vị trí nhìn trên ảnh (tổng = 1).
2. `mass_k` = tổng P trong elip đĩa k (có trọng số diện tích ô).
3. Nhân với inout (xác suất nhìn trong khung): `q_k = inout · mass_k`, `q_none = 1 − Σ_k q_k`. Model không có inout → inout = 1.
4. Frame `invalid` không đóng góp.

Mỗi trial: `p_look_k` = trung bình `q_k` trên các frame hợp lệ trong [t0, t1). Không có frame hợp lệ → rỗng.
Xuất cả: `p_look_k` (có "none", tổng = 1) và `p_look_plate_k = p_look_k / Σ_j p_look_j` (chỉ giữa các đĩa; tổng đĩa = 0 → rỗng).

Đây là **điểm tin cậy**, chưa hiệu chỉnh: p_look = 0.6 chưa chắc nghĩa là đúng 60% số lần. Hiệu chỉnh bằng dữ liệu `cue` (biết đĩa thật):
- `score` vẽ/ghi reliability: chia frame theo bin `q_max` (0–0.2, …, 0.8–1), mỗi bin tỉ lệ frame mà đĩa argmax q đúng đĩa được gọi; ghi ECE.
- `calibrate --run runs/<cue>`: fit temperature `T` (softmax(log(q + ε)/T) trên các đĩa + none) tối thiểu log-loss trên frame cue; ghi `T` vào `models/calibration.json`. Có file này thì UI và trials.csv dùng thêm cột `p_look_cal_k`.

Ví dụ (unit test): 1 frame, mass A = 0.6, B = 0.2, C = 0.0, inout = 0.9 → q = A 0.54, B 0.18, C 0, none 0.28; p_look_plate A 0.75, B 0.25, C 0.

- Món ở mỗi đĩa lấy từ trial (chế độ `collect`) → mọi số theo đĩa cũng hiện theo tên món.

## 2. Setup

- 1 webcam cố định thấy cả đầu người dùng và 3 đĩa trong cùng khung hình; chỉ 1 người trong khung.
- 3 đĩa không chồng nhau trong ảnh, xoè quạt khi nhìn từ đầu người dùng (không 2 đĩa trên cùng một tia).
- Món: thẻ giấy in ảnh/tên món đặt vào đĩa (đổi nhanh giữa các lượt) hoặc đồ ăn thật.
- Máy: Windows + NVIDIA. Device: cuda > cpu. Camera mở bằng `cv2.VideoCapture(index, cv2.CAP_DSHOW)` trên Windows (config `source.backend`).

## 3. Cấu trúc thư mục

```text
gazedemo/
  gazedemo/
    __main__.py          # python -m gazedemo <lệnh>
    cli.py               # argparse: doctor, bench, setup-aoi, live, cue, collect, offline, score, dataset, train, report
    types.py  config.py
    core/                # THUẦN: chỉ numpy + gazedemo.types
      aoi.py             # EllipseAOI, masks 64×64, assign_label()
      dwell.py           # samples → segments → dwells
      metrics.py         # compute(samples, t0, t1, cfg) → TrialResult (TTFD/DD/DC/p_look/...)
      features.py        # TrialResult (+ samples) → hàng feature theo (trial, đĩa)
      choice_model.py    # conditional logit numpy: fit / predict_proba / to_json / from_json
      eval.py            # baselines, CV theo trial, top-1, log-loss, confusion, bootstrap CI
    perception/
      gazelle_model.py   # load + infer(frame_rgb) → (heatmap 64×64 np.float32, inout float|None)
      fake.py            # FakeGaze theo kịch bản, cho test không GPU
    io/
      camera.py          # LatestFrameCamera (thread, 1 slot), VideoFileSource
      recorder.py        # ghi video.mp4 + frames.csv (thread riêng, mọi frame)
      logs.py            # samples.jsonl, dwells.csv, trials.csv, events.jsonl
    app/
      tracker.py         # GazeDwellTracker
      runner.py          # vòng chính dùng chung cho live / cue / collect / offline
      ui.py              # vẽ overlay + bảng
      setup_aoi.py  bench.py  doctor.py  synth.py
  configs/default.yaml
  configs/foods.yaml
  runs/                  # output mỗi buổi
  tests/
```

Luật kiến trúc (có test):
- `core/` không import cv2, torch, time, os, pathlib, json, csv.
- Một hàm `compute()` duy nhất cho live, cue, collect, offline, dataset.
- Timestamp sample = `time.monotonic()` lúc chụp frame (thread camera gán), không phải lúc Gazelle xong.
- Camera buffer 1 slot: Gazelle chậm hơn camera thì bỏ frame cũ. Recorder nhận **mọi** frame qua queue riêng để `offline` chạy lại đủ fps.
- `cv2.imshow` / `cv2.waitKey` chỉ ở main thread. Ghi file và video ở thread riêng.
- Mọi ngưỡng trong `configs/default.yaml`.

## 4. Xử lý

### 4.1 Gazelle (local, không head box)
- `model, transform = torch.hub.load("fkryan/gazelle", cfg.gazelle.name)`; `model.eval().to(device)`. Nếu người dùng đưa script Gazelle cũ, dùng đúng cách load trong đó.
- Input: BGR → RGB → PIL → `transform(img).unsqueeze(0).to(device)` = `[1, 3, 448, 448]`; `"bboxes": [[None]]`.
- Output: `out["heatmap"][0][0]` → `[64, 64]` (giá trị [0, 1]); `out["inout"][0][0]` (1 = nhìn trong khung; `None` nếu model không có nhánh inout).
- `torch.inference_mode()`; tuỳ chọn `torch.autocast("cuda", dtype=torch.float16)`; warm-up `cfg.gazelle.warmup` frame.
- Ô (i, j) có tâm `x = (j + 0.5)/64 · W`, `y = (i + 0.5)/64 · H` trên ảnh gốc. `gaze_xy` = tâm ô argmax.
- Exception khi infer → sample `label="invalid"`, log lỗi, không dừng chương trình.

### 4.2 Gán đĩa
Mỗi elip → mask `W_k` 64×64 = tỉ lệ diện tích ô nằm trong elip (lấy mẫu 4×4 điểm/ô). Elip chồng nhau → `setup-aoi` từ chối.

```python
def assign_label(H, inout, masks, cfg) -> tuple[str, dict[str, float]]:
    P = np.maximum(H, 0) ** cfg.gamma
    P = P / (P.sum() + 1e-9)
    mass = {k: float((P * W).sum()) for k, W in masks.items()}
    mass["none"] = max(0.0, 1.0 - sum(mass.values()))
    if inout is not None and inout < cfg.tau_inout:
        return "out", mass
    ranked = sorted(masks, key=lambda k: mass[k], reverse=True)
    best = ranked[0]
    second = mass[ranked[1]] if len(ranked) > 1 else 0.0
    if mass[best] >= cfg.tau_mass and mass[best] - second >= cfg.delta:
        return best, mass
    return "none", mass
```

### 4.3 Dwell (3 bước, đúng thứ tự)
Sample i phủ `[t_i, t_{i+1})`, sample cuối phủ `[t_n, t_end)`. Khoảng > `max_interval_s` bị cắt, phần thừa là `invalid`. Gộp khoảng liền nhau cùng nhãn thành segment.
1. Nối khoảng hở: đĩa X, rồi segment `none`/`invalid` tổng ≤ `max_gap_ms`, rồi X → một segment X. Không nối qua đĩa khác hay `out`.
2. Lọc nhấp nháy: segment đĩa < `min_dwell_ms` → `none`.
3. Nối khoảng hở lần nữa.

### 4.4 Metric trong [t0, t1)
- Cắt timeline theo cửa sổ (tracker giữ sample từ `t0 − lookback_s`); dwell còn < `min_dwell_ms` sau khi cắt → `none`.
- Theo đĩa: `ttfd_ms, dd_ms, dc, first_dwell_ms, longest_dwell_ms, mean_dwell_ms, share` (DD/tổng DD), `p_look`, `dd_last1s_ms`, `dd_last2s_ms` (DD trong 1 s/2 s cuối trước t1), `is_last` (đĩa của dwell cuối), `is_first`, `mean_dist_norm` (trung bình khoảng cách `gaze_xy` → tâm elip chia bán trục trung bình, trên sample hợp lệ).
- Chung: `first_aoi, last_aoi, none_ms, out_ms, invalid_ms, valid_ratio, n_samples, eff_hz`.
- Bất biến: Σ DD + none_ms + out_ms + invalid_ms = t1 − t0.
- Trial chỉ chốt khi có sample t ≥ t1 hoặc sau `finalize_timeout_s`; sample t ≥ t1 không dùng.

### 4.5 Ví dụ tính tay (unit test bắt buộc)
20 Hz, [0, 1.0) s, min_dwell 100, max_gap 150.
Nhãn: none 0.00–0.20 | A 0.20–0.50 | none 0.50–0.60 | A 0.60–0.70 | B 0.70–0.75 | C 0.75–1.00.

| Đĩa | DD | DC | TTFD |
| --- | --- | --- | --- |
| A | 500 | 1 | 200 |
| B | 0 | 0 | rỗng |
| C | 250 | 1 | 750 |

none_ms = 250, tổng 1000, first_aoi A, last_aoi C. Biến thể:
- A 0.20–0.50, none 0.50–0.70, A 0.70–0.80, C 0.80–1.00 → A 400/2/200; C 200/1/800.
- A 0.20–0.40, B 0.40–0.45, A 0.45–0.60, none sau → A 400/1.
- Đang nhìn A từ trước t0 → TTFD_A = 0; dwell vượt t1 bị cắt ở t1.
- Hai sample cách 1.0 s, max_interval 0.3 → 0.7 s invalid.
- p_look: xem ví dụ ở mục 1.1; trung bình nhiều frame = trung bình q từng frame.

## 5. Chế độ chạy (`python -m gazedemo <lệnh>`)

Mọi lệnh có camera đều: ghi `video.mp4` + `frames.csv` (nếu `record: true`), `samples.jsonl`, `events.jsonl` (mọi sự kiện trial với timestamp monotonic), `trials.csv`, `dwells.csv`, bản sao `config.yaml`, `aoi.json` vào `runs/<YYYYmmdd_HHMMSS>_<mode>/`.

| Lệnh | Làm gì |
| --- | --- |
| `doctor` | In: Python, torch, CUDA có không, tên GPU, opencv, mở được camera index nào, độ phân giải/fps thật, Gazelle load được không. Bước đầu tiên ở quán. |
| `bench --n 200` | Mean/p95 thời gian: đọc frame, Gazelle, gán đĩa; Hz cuối; ViT-L và ViT-B × fp32/fp16. Ghi `bench.md`. |
| `setup-aoi` | Frame live; lần lượt kéo chuột vẽ hộp A, B, C → elip nội tiếp; +/− margin; U hoàn tác; R chụp lại frame; Enter lưu `aoi.json` (+ `aoi_ref.png`). Vẽ tia từ điểm người dùng click (đầu) đến tâm từng đĩa, in góc giữa các tia; cảnh báo nếu < 15°. |
| `live` | Trial bằng phím. S bắt đầu · E kết thúc · 1/2/3 gán nhãn đĩa cho trial vừa xong · H bật/tắt heatmap · Q thoát. |
| `cue` | Nhìn theo hiệu lệnh, tự gán nhãn: `n_rep` lần mỗi đĩa, thứ tự ngẫu nhiên (seed trong config). Mỗi lượt: màn hình hiện chữ lớn "NHÌN A" + tiếng bíp; trial = [t_cue + `settle_s`, t_cue + `cue_s`), label = đĩa được gọi. Hết lượt in luôn kết quả `score`. |
| `collect` | Thu dữ liệu chọn món (mục 6). |
| `offline --run runs/<id>` | Chạy lại pipeline trên `video.mp4` theo timestamp trong `frames.csv`, dùng `events.jsonl` để dựng lại trial; ghi vào `runs/<id>/offline/` cùng định dạng. `--stride k` xử lý mỗi k frame. |
| `score --run runs/<id>` | Với trial có label: dự đoán = đĩa có DD lớn nhất (tất cả 0 → "none"). In và ghi `score.md`: số trial, accuracy, confusion 3×4 (A/B/C/none), accuracy theo p_look argmax, bảng reliability + ECE theo frame (mục 1.1). |
| `calibrate --run runs/<cue>` | Fit temperature T cho p_look từ dữ liệu cue → `models/calibration.json` (mục 1.1). Làm cùng mốc M4. |
| `dataset --runs runs/ --out data/gaze_choice.csv` | Gom mọi run `collect` (và `offline/` nếu có, ưu tiên offline) thành dataset dạng dài (mục 7.1). |
| `train --data data/gaze_choice.csv --out models/` | CV + baseline + conditional logit (mục 7.2); ghi `models/model.json`, `models/eval.json`, `models/report.md`. |
| `report --out report/` | Gom `bench.md`, `score.md` mới nhất, thống kê dataset, `models/report.md`, 1 ảnh chụp frame có overlay, 1 biểu đồ timeline nhãn của 1 trial (matplotlib nếu có, không có thì bỏ qua) → `report/SUMMARY.md`. |

### 5.1 UI (live, cue, collect)
- Video + 3 elip (màu riêng; đĩa đang nhìn viền dày), điểm argmax, heatmap phủ mờ (H).
- Góc trên: Hz xử lý, latency (ms), inout, nhãn hiện tại, chế độ, trial id, thời gian trial.
- Bảng mỗi đĩa, cập nhật mỗi sample (provisional khi trial đang chạy):

```text
Trial 7 (collect)   t = 3.2 s   14.8 Hz   valid 97%
đĩa  món      TTFD   DD    DC   p_look  p_choice
A    cơm       350  1800    2    0.58     0.71
B    trứng    2100   900    1    0.30     0.22
C    rau         —     0    0    0.12     0.07
```

- p_choice chỉ hiện khi `--model models/model.json` được truyền và model load được.
- Cửa sổ có thể phóng to; chữ đủ lớn để quay màn hình.
- Kết thúc trial: in 1 dòng tóm tắt ra console.

## 6. Thu dữ liệu chọn món (`collect`)

`python -m gazedemo collect --participant P01 --foods configs/foods.yaml --trials 30`

Một người làm một mình nên luồng phải điều khiển hoàn toàn bằng bàn phím:
1. Đầu buổi: với mỗi món trong `foods.yaml`, nhập mức thích 1–5 (phím số). Lưu vào `session.json`.
2. Sinh lịch: mỗi lượt chọn 3 món khác nhau và xếp vào A/B/C sao cho mỗi món xuất hiện ở mỗi đĩa số lần gần bằng nhau (lịch cân bằng, seed cố định, lưu `schedule.csv`).
3. Mỗi lượt:
   - Màn hình hiện "Lượt 7/30 — Đặt: A = cơm · B = trứng · C = rau". Người dùng đặt thẻ, ngồi về vị trí, nhấn **Space**.
   - Đếm ngược 1 s (không tính), rồi bíp → **t0**. Màn hình chỉ hiện chữ "Hãy nhìn và chọn món bạn muốn" (không hiện số liệu để không ảnh hưởng).
   - Sau `collect.trial_s` giây: bíp → **t1** (`end_trial(t_beep)`). Gaze sau t1 không dùng.
   - Màn hình hỏi "Bạn chọn đĩa nào? 1 / 2 / 3" (X = bỏ lượt). Ghi `chosen_plate`, `chosen_food`.
   - Hiện kết quả lượt đó (bảng ở 5.1) 2 s rồi sang lượt sau.
   - Phím: Space bắt đầu · 1/2/3 chọn · X bỏ · P tạm dừng · Q lưu và thoát (thoát giữa chừng vẫn giữ các lượt đã xong).
4. `trials.csv` thêm cột: `participant, food_A, food_B, food_C, chosen_plate, chosen_food, liking_A, liking_B, liking_C, skipped`.
5. Bíp trên Windows: `winsound.Beep`; nơi khác: in `\a`. Bọc trong try/except.

## 7. Dataset và classifier

### 7.1 Dataset (dạng dài)
Mỗi dòng = một cặp (trial, đĩa), bỏ lượt `skipped`:
`run_id, participant, trial_id, plate, food, liking, chosen (0/1)` + feature:
- G1: `ttfd_ms` (rỗng → điền `t1 − t0` và thêm cờ `never_looked` = 1), `dd_ms`, `dc`
- G2: `share`, `longest_dwell_ms`, `mean_dwell_ms`, `p_look`
- G3: `dd_last1s_ms`, `dd_last2s_ms`, `is_last`, `is_first`
- G4: `mean_dist_norm`
- Chất lượng: `valid_ratio`, `eff_hz` (không dùng làm feature, dùng để lọc: bỏ trial `valid_ratio < cfg.dataset.min_valid_ratio`).
Ghi kèm `data/gaze_choice_trials.csv` (một dòng/trial) và in thống kê: số trial, phân bố đĩa được chọn, phân bố món được chọn.

### 7.2 Train và đánh giá
- Mô hình chính: **conditional logit** — P(chọn đĩa k | trial) = softmax_k(wᵀx_k). Numpy, L2 (`train.l2`), tối ưu bằng gradient descent/L-BFGS tự viết hoặc `scipy.optimize` nếu có; chuẩn hoá feature bằng mean/std từ tập train của fold. Bộ feature theo config: `g1`, `g1g2`, `g1g2g3`, `all`, `liking_only`, `all+liking`.
- Baseline: chance (1/3), argmax DD, argmax p_look, đĩa nhìn cuối, đĩa nhìn đầu, dwell ≥ T (đĩa đầu tiên có một dwell ≥ `T_ms`, không có → argmax DD), argmax liking.
- CV: có ≥ 2 participant → leave-one-participant-out; chỉ 1 participant (trường hợp đợt này) → K-fold theo trial (`train.k_folds`, mọi đĩa của một trial cùng một fold), lặp với 5 seed.
- Metric: top-1 accuracy, log-loss (baseline cứng dùng xác suất 1−ε cho đĩa được chọn), confusion theo đĩa, CI 95% bootstrap theo trial (1000 lần).
- Output: `models/eval.json`, `models/report.md` (bảng mô hình × metric, sắp theo accuracy, có cột n_trial; câu cảnh báo cố định: "Dữ liệu 1 người, người thu dữ liệu cũng là người thiết kế thí nghiệm; kết quả chỉ chứng minh pipeline chạy, chưa phải bằng chứng tổng quát."), `models/model.json` (model `all` train trên toàn bộ dữ liệu: tên feature, mean/std, trọng số, ngày train, n_trial).
- Ít hơn `train.min_trials` trial → vẫn chạy, nhưng report ghi rõ "quá ít dữ liệu".

### 7.3 Dữ liệu giả lập (để test khi chưa có data)
`python -m gazedemo synth --participants 1 --trials 60 --out runs_synth/` sinh các run giả có cùng định dạng thật (samples.jsonl, trials.csv, events.jsonl, session.json) bằng FakeGaze: đĩa được chọn có xác suất nhận nhiều dwell dài hơn, và nhìn cuối nhiều hơn. Dùng cho test end-to-end `dataset → train`.

## 8. Config mặc định
Xem `configs/default.yaml`. Mọi số là placeholder.

## 9. Test (pytest; trừ mark `gpu` và `camera`, chạy không cần GPU/camera)

| File | Kiểm tra |
| --- | --- |
| test_dwell.py | Ví dụ 4.5 + biến thể; bất biến tổng thời gian; rỗng; 1 sample |
| test_aoi.py | Mask trong [0, 1], tổng ≤ 1; Gaussian ở tâm B → B; giữa A và B → none; inout thấp → out; heatmap → px |
| test_metrics.py | p_look; dd_last1s/2s; is_last/is_first; mean_dist_norm |
| test_tracker.py | Chốt trial muộn; loại sample t ≥ t1; snapshot provisional |
| test_e2e_fake.py | FakeCamera + FakeGaze kịch bản → runner chế độ live không cửa sổ → trials.csv đúng, sai ≤ 1 sample |
| test_offline_equiv.py | Run giả có video ngắn + frames.csv + events → offline cho cùng trials.csv như live khi FakeGaze tất định |
| test_collect_schedule.py | Lịch cân bằng món × đĩa; 3 món khác nhau mỗi lượt |
| test_dataset_train.py | synth → dataset → train: chạy trọn; conditional logit > chance trên dữ liệu giả có tín hiệu; model.json load lại cho cùng xác suất |
| test_choice_model.py | Gradient đúng (so sai phân hữu hạn); xác suất mỗi trial cộng = 1 |
| test_architecture.py | Luật import của core/ |
| test_gazelle_smoke.py (gpu) | Load model, 1 ảnh: heatmap (64, 64) |

## 10. Thứ tự làm (mỗi mốc ra được thứ cho xem)

| Mốc | Nội dung | Xong khi |
| --- | --- | --- |
| M0 | `doctor`, `perception/gazelle_model.py`, `bench` | doctor báo CUDA + camera OK; có `bench.md` |
| M1 | types, config, core (aoi, dwell, metrics) + test | test_dwell/aoi/metrics/architecture pass |
| M2 | `setup-aoi` | vẽ 3 elip, lưu và load lại được |
| M3 | camera, recorder, logs, tracker, runner, ui, `live`, `score` | chạy 5 phút latency không tăng dần; test_tracker, test_e2e_fake pass |
| M4 | `cue` | 15 lượt chạy trọn, in score |
| M5 | `collect` | 5 lượt thử chạy trọn; test_collect_schedule pass |
| M6 | features, choice_model, eval, `synth`, `dataset`, `train`; `live/collect --model` | test_dataset_train, test_choice_model pass |
| M7 | `offline`, `report` | test_offline_equiv pass; có `report/SUMMARY.md` |

Ưu tiên khi thiếu thời gian: M0 → M1 → M2 → M3 → M4 → M5 (để **thu được data** trước khi hết giờ) → M6 → M7. M6 không cần camera/GPU nên có thể làm ở nhà sau trên data đã thu.
