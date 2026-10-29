# Đặc tả đầy đủ module gazedwell: đĩa đang nhìn + DD/DC/TTFD (Gazelle, real-time)

> **Đọc trước:** đây là thiết kế đầy đủ (bản 28/10/2026, xuất từ Claude Docs) cho cả các mốc sau.
> Bản demo hiện tại chỉ làm phần trong `docs/SPEC.md` (M0–M3). Khi hai file khác nhau, **`SPEC.md` thắng** cho đợt này:
> - Repo/package tên `gazedemo` (bản đầy đủ gọi là `gazedwell`).
> - `aoi.mode: manual` (vẽ tay); chưa làm ArUco, `aoi_version`, cập nhật AOI giữa bữa.
> - `head.mode: none` (bbox `None`); chưa làm MediaPipe, `fixed`.
> - Recorder/offline, validate (lệnh `cue`), collect, dataset, train **có làm** nhưng theo phiên bản rút gọn trong SPEC.md (1 người, K-fold theo trial, thêm p_look/p_choice). Số mốc trong SPEC.md khác số mốc ở đây.
>
> File này để hiểu **vì sao** thiết kế như vậy và **sau này mở rộng thế nào**: viết code demo sao cho các mục ở đây thêm vào được mà không phải viết lại (core thuần, một hàm `compute()`, timestamp lúc chụp frame, log đủ `mass`/`gaze_xy`/`inout`).

## 1. Mục tiêu và định nghĩa

Module chạy liên tục từ camera: mỗi frame → Gazelle → nhãn "đang nhìn đĩa nào" → gom thành dwell → xuất cho từng đĩa 3 số trong một cửa sổ thời gian (trial). Bỏ hẳn kiểu nhấn Space để ra một ảnh có chấm đỏ.

Output bắt buộc cho mỗi đĩa A, B, C (số đĩa cấu hình được):

| Metric | Ký hiệu | Định nghĩa | Đơn vị |
| --- | --- | --- | --- |
| Dwell duration | DD | Tổng thời gian các dwell trên đĩa, cắt theo cửa sổ | ms |
| Dwell fixation | DC | Số lần ghé (số dwell) trên đĩa trong cửa sổ | lần |
| Time to first dwell | TTFD | Từ đầu cửa sổ đến lúc bắt đầu dwell đầu tiên trên đĩa; `null` nếu không ghé | ms |

- **Dwell** = một lần nhìn liên tục vào cùng một đĩa, từ lúc vào đến lúc rời, dài ít nhất `min_dwell_ms` (mục 6).
- **"Dwell fixation" = DC (dwell count)**, khớp nhóm feature G1 = TFD/DD/DC trong plan dự án. Không đếm fixation thật: saccade chỉ vài chục ms, ngắn hơn khoảng cách giữa 2 lần Gazelle chạy, nên chỉ đo được dwell.
- Output phụ (phục vụ feature G2/G3): dwell đầu tiên dài bao nhiêu, dwell dài nhất, tỉ lệ DD của đĩa / tổng DD, đĩa nhìn đầu, đĩa nhìn cuối, tỉ lệ thời gian hợp lệ, tần số xử lý thực tế.
- Ngoài phạm vi: module đo "đang nhìn đĩa nào", không đo "muốn ăn món nào". Classifier chọn món dùng feature từ đây (M6–M7).

## 2. Setup phần cứng và môi trường

Một camera RGB cố định thấy cả đầu người dùng lẫn 3 đĩa: Gazelle ước lượng điểm nhìn trên ảnh cả cảnh, nên người và đĩa phải cùng nằm trong ảnh.

**Camera**
- RealSense D435i (chỉ luồng màu) hoặc webcam, gắn tripod/kẹp chắc, không chạm vào trong buổi.
- 1280×720, 30 fps. Gazelle co ảnh về 448×448.
- Khoá auto-exposure, auto white balance, auto-focus nếu camera cho phép.
- Góc chéo phía trước người dùng (draft rig: ~50 cm, chếch với pháp tuyến của mặt phẳng mặt người khoảng 5-10 độ, chưa test).

**Đặt đĩa**
- 3 đĩa không chồng nhau trong ảnh; khe hở giữa các đĩa càng rộng càng tốt (heatmap 64×64 bị nhoè).
- Từ đầu người dùng trong ảnh, 3 đĩa nằm theo 3 hướng khác nhau (xoè quạt). Hai đĩa trên cùng một đường thẳng từ đầu đi ra → model phải đoán xa/gần dọc tia và dễ nhầm. Tool `setup-aoi` vẽ tia đầu → tâm đĩa và in góc giữa chúng.
- Đánh dấu vị trí đĩa và chân camera bằng băng dính để lặp lại giữa các buổi.
- Mặt đủ sáng, không ngược sáng.

**Máy chạy:** laptop RTX 4050 (CUDA). FPS chưa đo → M0 là benchmark. Device: cuda > mps > cpu.

## 3. Kiến trúc

Hai luồng: thread camera giữ frame mới nhất; vòng chính chạy (head box) → Gazelle → gán đĩa → tracker → UI/log. Phần tính toán (gán đĩa, dwell, metric) là hàm thuần, dùng chung cho live, offline và validation. Đường offline: video đã ghi được phát lại qua đúng các bước của vòng chính.

**Nguyên tắc bắt buộc**
- `core/` chỉ import numpy và `types`; không cv2, torch, file, thời gian hệ thống.
- Một hàm duy nhất tính metric (`core.metrics.compute`) cho live, offline, validation và dữ liệu training.
- Thời gian của sample = `time.monotonic()` lúc chụp frame, không phải lúc Gazelle chạy xong.
- Buffer camera 1 slot: Gazelle chậm hơn camera thì bỏ frame cũ, không xếp hàng. Recorder (sau này) ghi đủ mọi frame để offline chạy lại đủ 30 fps.
- `cv2.imshow` chạy ở main thread.
- Mọi ngưỡng nằm trong config.

Cấu trúc đầy đủ (bản demo chỉ có một phần):

```text
pkg/
  types.py  config.py
  core/        aoi.py  dwell.py  metrics.py
  perception/  head.py  gazelle_model.py  fake.py
  io/          camera.py  recorder.py  logs.py
  app/         tracker.py  live.py  setup_aoi.py  offline.py  bench.py  validate.py  collect.py
  cli.py
configs/default.yaml  models/  runs/  tests/
```

**Ghép vào repo dự án `feedbot` sau này:** `core/aoi`, `dwell`, `metrics` gộp vào `core/aoi`, `core/events`, `core/features` (G1) của feedbot; Gazelle điền STUB B5 `gaze_gazelle`; camera vào `adapters`; tracker/live/setup vào `app/`. AOI của feedbot ở toạ độ mặt bàn (tia đầu ∩ bàn); AOI ở đây ở toạ độ ảnh → thêm loại AOI mới chứ không thay. Khi lên ROS, nội dung một dòng `trials.csv` (mỗi đĩa DD/DC/TTFD + timestamp) thành message kiểu `/gaze/plate_metrics`.

## 4. Kiểu dữ liệu

Dataclass bất biến; thời gian bằng giây (monotonic), metric xuất ra bằng ms.

```python
# label của 1 sample: "A","B","C",... | "none" | "out" | "invalid"
@dataclass(frozen=True)
class EllipseAOI:
    name: str; cx: float; cy: float; rx: float; ry: float; angle_deg: float = 0.0

@dataclass(frozen=True)
class Sample:
    t: float                   # s, time.monotonic() lúc chụp frame
    frame_idx: int
    aoi_version: int           # (bản đầy đủ) tăng mỗi lần AOI đổi
    label: str
    mass: dict[str, float]     # khối heatmap trong từng AOI + "none"; {} nếu invalid
    gaze_xy: tuple[float, float] | None
    inout: float | None
    head_box: tuple[float, float, float, float] | None
    quality: str               # "ok" | "held_box" | "no_face" | "no_bbox"
    latency_ms: float

@dataclass(frozen=True)
class Dwell:
    aoi: str; t_start: float; t_end: float; n_samples: int

@dataclass(frozen=True)
class AOIMetrics:
    dd_ms: float; dc: int; ttfd_ms: float | None
    first_dwell_ms: float | None; longest_dwell_ms: float | None; share: float | None

@dataclass(frozen=True)
class TrialResult:
    trial_id: int; t0: float; t1: float
    per_aoi: dict[str, AOIMetrics]
    first_aoi: str | None; last_aoi: str | None
    none_ms: float; out_ms: float; invalid_ms: float
    valid_ratio: float; n_samples: int; eff_hz: float
    provisional: bool
    label: str | None = None
    meta: dict = field(default_factory=dict)
```

## 5. Xử lý mỗi frame

### 5.1 Head box
- Mặc định (chốt 28/10): `head.mode: none` — bbox `None` cho Gazelle; giả định khung hình chỉ có người bệnh (README Gazelle: chạy được không cần bbox khi cảnh có 1 người).
- Phương án so sánh sau này: `fixed` (hộp đầu vẽ tay một lần) và `mediapipe` (MediaPipe Tasks `FaceDetector`, nới box theo `expand`, dời lên `shift_up`, EMA, giữ box cũ tối đa `hold_s`). Dùng khi có người chăm sóc lọt vào khung.

### 5.2 Gazelle
- Model mặc định `gazelle_dinov2_vitl14_inout`; đổi `gazelle_dinov2_vitb14_inout` nếu quá chậm.
- `model, transform = torch.hub.load("fkryan/gazelle", name)`; `model.eval().to(device)`.
- Input: BGR → RGB → PIL → `transform(img).unsqueeze(0)` = `[1, 3, 448, 448]`; `bboxes = [[box_chuẩn_hoá hoặc None]]`.
- Output: `out["heatmap"][0][0]` = `[64, 64]`, giá trị [0, 1]; `out["inout"][0][0]` (1 = nhìn trong khung; chỉ bản `_inout`).
- `torch.inference_mode()`; tuỳ chọn fp16 autocast trên CUDA; warm-up 5 frame.
- Ô (i, j) có tâm `x = (j + 0.5)/64 · W`, `y = (i + 0.5)/64 · H`. Kiểm: argmax trùng chấm đỏ của script cũ, sai tối đa 1 ô.

### 5.3 Gán nhãn đĩa
Mask `W_k` 64×64 = tỉ lệ diện tích ô nằm trong elip (lấy mẫu 4×4/ô, numpy). Elip không được chồng nhau.

```python
def assign_label(H, inout, quality, masks, cfg):
    if quality == "no_face":
        return "invalid", {}
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

- Vì sao dùng khối heatmap thay vì chấm đỏ: heatmap chia đôi giữa 2 đĩa → `none` thay vì đoán bừa; và `mass` theo đĩa chính là p_gaze mềm cho công thức fusion trong plan dự án.
- `assign.method: argmax` là phương án so sánh. Luôn log cả `mass` lẫn `gaze_xy` để so 2 cách offline.
- `tau_mass`, `delta`, `tau_inout`, `gamma` là placeholder; chỉnh bằng dữ liệu validation.

## 6. Dwell và metric

Dwell dựng lại từ toàn bộ sample mỗi lần cần số (không cộng dồn), nên live, offline, validation cho cùng kết quả.

### 6.1 Từ sample thành dòng thời gian
- Sample i phủ `[t_i, t_{i+1})`; sample cuối phủ `[t_n, t_end)` (`t_end` = cuối cửa sổ hoặc "bây giờ").
- Khoảng dài hơn `max_interval_s` bị cắt; phần thừa là `invalid`.
- Gộp khoảng liền nhau cùng nhãn thành segment.

### 6.2 Dựng dwell: 3 bước, đúng thứ tự
1. **Nối khoảng hở:** đĩa X, rồi segment `none`/`invalid` tổng ≤ `max_gap_ms`, rồi lại X → một segment X (tính cả khoảng hở). Không nối qua đĩa khác, không nối qua `out`.
2. **Lọc nhấp nháy:** segment đĩa < `min_dwell_ms` → `none`.
3. **Nối khoảng hở lần nữa** (A–B nhấp nháy–A thành A–none–A).

Tracker giữ thêm sample từ `t0 − lookback_s` để nhận ra dwell đang diễn ra từ trước `t0`.

### 6.3 Metric trong cửa sổ [t0, t1)
- Cắt theo cửa sổ; dwell còn lại < `min_dwell_ms` → `none`.
- DD_k, DC_k, TTFD_k như mục 1; đang nhìn sẵn k lúc t0 → TTFD_k = 0.
- Bất biến: tổng DD + none_ms + out_ms + invalid_ms = độ dài cửa sổ.
- `valid_ratio` = 1 − invalid_ms / độ dài; `eff_hz` = số sample / độ dài.

### 6.4 Ví dụ tính tay (unit test)
20 Hz, [0, 1.0) s, min_dwell 100, max_gap 150. Nhãn: none 0–0.20 | A 0.20–0.50 | none 0.50–0.60 | A 0.60–0.70 | B 0.70–0.75 | C 0.75–1.00.
Bước 1 nối A → A [0.20, 0.70); bước 2 B (50 ms) → none. Kết quả: A 500/1/200; B 0/0/null; C 250/1/750; none 250.
Biến thể: (a) khoảng hở 200 ms → A DC 2; (b) A–B(50)–A → B bị lọc, A nối, DC 1; (c) dwell từ trước t0 → TTFD 0, dwell vượt t1 bị cắt; (d) 2 sample cách 1.0 s, max_interval 0.3 → 0.7 s invalid.

## 7. Trial, hàm Python và chế độ chạy (tất cả local)

| Chế độ | t0 | t1 | Dùng khi |
| --- | --- | --- | --- |
| Trial bằng phím | nhấn S | nhấn E hoặc t0 + `trial_s` | thử tay, demo |
| Trial theo sự kiện | cue / robot bắt đầu detect món | bíp | protocol 2a, robot |
| Cửa sổ trượt | bây giờ − `window_s` | bây giờ | runtime robot |

```python
class GazeDwellTracker:
    def update(self, sample: Sample) -> None
    def start_trial(self, t0=None, trial_id=None, meta=None) -> int
    def end_trial(self, t1=None, label=None) -> None
    def poll_finished(self) -> list[TrialResult]
    def snapshot(self, t_now) -> TrialResult | None        # provisional=True
    def window(self, window_s, t_now) -> TrialResult       # cửa sổ trượt
```

- "API" ở đây là hàm Python gọi trong cùng chương trình, không phải dịch vụ mạng.
- Chốt trial muộn: chỉ chốt khi có sample t ≥ t1 (hoặc sau `finalize_timeout_s`); sample t ≥ t1 không dùng ("chỉ dùng gaze trước bíp").
- `label` = đĩa thật sự được chọn/cue (phím 1/2/3 hoặc protocol truyền vào).

## 8. Output
- Cửa sổ live: video + elip (đĩa đang nhìn viền dày), điểm argmax, heatmap (H); góc trên Hz/latency/inout/nhãn/thời gian trial; bảng DD/DC/TTFD từng đĩa. Phím S/E/1–3/H/Q.
- File trong `runs/<YYYYmmdd_HHMMSS>_<participant>/`: `config.yaml`, `aoi.json`, (sau này `video.mp4` + `frames.csv`), `samples.jsonl`, `dwells.csv`, `trials.csv`, `trials.jsonl`.
- Cột `trials.csv`: `trial_id, t0, t1, dur_ms, label, A_dd_ms, A_dc, A_ttfd_ms, B_..., C_..., first_aoi, last_aoi, none_ms, out_ms, invalid_ms, valid_ratio, eff_hz, n_samples`. TTFD null → ô trống.
- Ghi file qua thread riêng, không chặn vòng chính. Video mặt người tham gia theo thủ tục consent.

## 9. Công cụ phụ
- **setup-aoi:** bản đầy đủ dùng ArUco (`DICT_4X4_50`, id 0/1/2 = A/B/C, marker dán ở vành đĩa): đầu bữa detect → đề xuất elip → người chăm sóc nhấn Enter; giữa bữa, khi không có trial mở, detect lại mỗi `refresh_s`, đĩa dịch quá `move_px` thì dựng AOI mới, tăng `aoi_version`; trial đang mở không bao giờ đổi AOI. `aoi.mode: manual` (vẽ tay) là dự phòng — và là chế độ của bản demo.
- **offline (M4):** chạy lại pipeline trên video đã ghi ở đủ fps camera; `--reuse-samples` tính lại dwell/metric từ `mass` đã log với config mới.
- **bench (M0):** thời gian từng bước, mean/p95, ViT-L/ViT-B × fp32/fp16. Độ phân giải thời gian của DD/TTFD ≈ 1/Hz.
- **validate (M5):** nhìn theo hiệu lệnh A/B/C ngẫu nhiên, mỗi lệnh `cue_s`, bỏ `settle_s` đầu; confusion matrix, accuracy, Cohen's κ; tỉ lệ trial mà đĩa có DD lớn nhất = đĩa được cue; quét ngưỡng offline.

## 10. Config đầy đủ (placeholder)

```yaml
source: {kind: opencv, device: 0, width: 1280, height: 720, fps: 30, record: true, save_heatmaps: false}
aoi: {mode: aruco, aruco_dict: DICT_4X4_50, marker_ids: {A: 0, B: 1, C: 2}, refresh_s: 2.0, move_px: 15,
      file: aoi.json, names: [A, B, C], margin: 0.10}
head: {mode: none, expand: 1.5, shift_up: 0.10, ema_alpha: 0.5, hold_s: 0.5, on_lost: invalid}
gazelle: {name: gazelle_dinov2_vitl14_inout, device: auto, fp16: true, warmup: 5}
assign: {method: mass, gamma: 1.0, tau_mass: 0.35, delta: 0.10, tau_inout: 0.5}
dwell: {min_dwell_ms: 100, max_gap_ms: 150, gap_labels: [none, invalid], max_interval_s: 0.3, lookback_s: 2.0, buffer_s: 10.0}
trial: {trial_s: null, window_s: 6.0, finalize_timeout_s: 1.0}
validate: {cue_s: 4.0, settle_s: 1.0, n_rep: 5}
output: {root: runs, participant: P00}
```

## 11. Test
test_dwell (ví dụ 6.4 + biến thể + bất biến), test_aoi, test_tracker, test_equivalence (tracker từng sample = compute một lần; offline tái tạo đúng live), test_e2e_fake, test_architecture, test_gazelle_smoke (gpu).

## 12. Mốc đầy đủ
- M0 benchmark · M1 core · M2 setup-aoi · M3 live — **bản demo dừng ở đây** (+ `score_trials.py`).
- M4 recorder + offline · M5 validate.
- M6 runner thu dữ liệu protocol 2a: `collect --participant --foods`; đầu buổi nhập 6 món + mức thích; mỗi lượt xếp 3/6 món vào A/B/C cân bằng; Space → start_trial; bíp → end_trial(t_beep); phím 1/2/3 nhập đĩa được chọn; `trials.csv` thêm `food_A, food_B, food_C, chosen_plate, chosen_food, aoi_version`.
- M7 dataset + train/eval: mỗi dòng một (trial, đĩa) với feature G1 DD/DC/TTFD; G2 tỉ lệ DD, dwell dài nhất/trung bình; G3 DD trong 1 s và 2 s cuối trước bíp, có phải đĩa nhìn cuối; G4 khoảng cách điểm nhìn–tâm đĩa (chia bán kính), `mass` trung bình. Baseline (chance, argmax DD, nhìn cuối, nhìn đầu, dwell > T, chỉ mức thích) + conditional logit; LOSO; top-1, log-loss, ECE, accuracy–coverage, CI bootstrap theo người. Test trên dữ liệu giả lập 5 người.

## 13. Rủi ro
- README Gazelle: bản `_inout` train GazeFollow → fine-tune VideoAttentionTarget. Giả thuyết: rig này (mặt gần camera, đĩa nhỏ cạnh nhau) khác xa dữ liệu train → "tương đối ok" chưa chắc giữ khi đổi người/vị trí đĩa. Validate trả lời bằng số.
- Gazelle xử lý từng frame độc lập; nếu κ thấp, thử làm mượt theo thời gian trên `mass` (vd. HMM) trước khi đổi model.
- Đĩa gần thẳng hàng khi nhìn từ đầu → nhầm đĩa kề.
- Tay robot che đĩa/mặt → `invalid`/`none`.
- Bỏ head box: nếu người chăm sóc lọt vào khung, Gazelle có thể đo nhầm người.

Nguồn: Gaze-LLE README (github.com/fkryan/gazelle); Ryan et al., "Gaze-LLE", CVPR 2025, arXiv 2412.09586.
