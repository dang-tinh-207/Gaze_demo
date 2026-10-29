# Gaze_demo — which plate is the user looking at?

A camera watches a person and three plates (A, B, C). Gazelle (Gaze-LLE, ViT-L) runs locally on an NVIDIA GPU and estimates where the person is looking, frame by frame. `gazedemo` turns that into gaze metrics per plate and food, and it can train a small classifier that predicts which food the person will choose.

For each round (trial) and each plate, it outputs:

| Metric | Meaning |
| --- | --- |
| **TTFD** | Time to first dwell: ms from the start of the round to the first look at the plate (empty if never looked at) |
| **DD** | Dwell duration: total ms spent looking at the plate |
| **DC** | Dwell count: number of separate visits to the plate |
| **p_look** | Probability that the person is looking at the plate (averaged over the round; plates + "none" sum to 1) |
| **p_choice** | Probability that the person chooses that plate's food (only after a model has been trained) |

```
Gaze_demo/
├── gazedemo/gazedemo/          ← the project (run every command from here)
│   ├── gazedemo/               ← Python package
│   ├── configs/default.yaml    ← all settings and thresholds
│   ├── configs/foods.yaml      ← food list used by `collect`
│   ├── docs/                   ← specification and background
│   ├── tests/
│   └── runs/                   ← one folder per session (created automatically)
└── gazelle-main (1)/gazelle-main/   ← Gazelle source code + checkpoints/ (ViT-L, ViT-B)
```

---

## 1. Hardware setup

**You need:**
- **A computer with an NVIDIA GPU.** It was tested on an RTX 4060 Laptop GPU (8 GB): Gazelle ViT-L runs at about 58 ms per frame (≈ 17 Hz) in fp16 and uses about 3.5 GB of GPU memory. Without a GPU it runs on the CPU but is far too slow for live use.
- **One fixed RGB camera:**
  - *Intel RealSense D435i:* only the colour stream is used. Connect it to a **USB 3** port; on USB 2 the colour stream is limited to 15 fps.
  - *Any USB webcam:* 1280×720 at 30 fps is ideal.
- **Three plates** (paper plates are fine) labelled A, B, C, and food cards or real food that match `configs/foods.yaml`.
- **A tripod or clamp** for the camera, and tape to mark where the camera and plates go.

**Placement (important for accuracy):**
1. The camera must see **the person's head and all three plates in the same image**. Gazelle estimates gaze from the whole scene, so both must be in view.
2. Put the camera in front of the person, slightly to one side, and fix it firmly. Don't touch it during a session.
3. Spread the plates **like a fan as seen from the person's head**: no two plates on the same line of sight. `setup-aoi` measures the angle between plates and warns if it is under 15°.
4. Keep gaps between the plates in the image; plates must not overlap.
5. **Light the face and the plates well**, and avoid backlight such as a window behind the person. Dark images make the gaze estimate unreliable.
6. Only **one person** should be in the image.
7. Mark the camera and plate positions with tape so later sessions can repeat the setup.

**Linux + RealSense note:** if Intel's `librealsense2-dkms` package is installed and Secure Boot is on, the camera driver won't load (`ls /dev/video*` shows nothing) until its signing key is enrolled. To enroll it:
1. Run `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` and choose a one-time password.
2. Reboot. On the blue screen choose **Enroll MOK → Continue → Yes**, enter the password, then reboot.
3. Run `sudo modprobe uvcvideo` if `/dev/video*` still doesn't appear.

Don't run the ROS `realsense2_camera` node at the same time: only one program can use the camera.

---

## 2. Library requirements

| Package | Notes |
| --- | --- |
| Python | 3.10 or 3.11 |
| torch, torchvision | Install the **CUDA** build that matches your driver (see pytorch.org). Tested with torch 2.5.1 + CUDA 11.8. |
| timm | Transformer blocks used by Gazelle |
| opencv-python | Camera, video, window |
| numpy, pyyaml | |
| pillow | Comes with torchvision; used for on-screen text |
| pytest | To run the tests |
| scipy, matplotlib | Optional; matplotlib draws the timeline in `report` |

scikit-learn is **not** needed. Gazelle's code imports it, but gazedemo works around that.

**Install (conda, Linux or Windows):**
```bash
conda create -n gazelle python=3.10 -y
conda activate gazelle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm opencv-python numpy pyyaml pytest scipy matplotlib
```

**Or with a venv on Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python numpy pyyaml pytest scipy matplotlib
```

**Model weights:**
- **Gazelle decoder checkpoints** are already in this repo, in `gazelle-main (1)/gazelle-main/checkpoints/`.
- **The DINOv2 backbone** is downloaded automatically the **first** time Gazelle loads: about 1.2 GB for ViT-L, so internet is needed once. After that it is cached in `~/.cache/torch/hub/`.
- **If the `gazelle-main` folder is missing,** gazedemo falls back to downloading Gazelle through `torch.hub` (`fkryan/gazelle`).

**Check the installation:**
```bash
cd gazedemo/gazedemo
python -m pytest -m "not gpu and not camera" -q    # should end with "passed", no failures
```

---

## 3. Running guidance

Every command runs in a terminal, from the project folder:
```bash
conda activate gazelle
cd Gaze_demo/gazedemo/gazedemo
```

**All terminal commands, in order** (each step is explained below):
```bash
python -m gazedemo doctor                                   # 0. check GPU, camera, Gazelle
python -m gazedemo bench --n 200                            # 1. measure speed (optional)
python -m gazedemo setup-aoi                                # 2. draw the 3 plates
python -m gazedemo live                                     # 3. live check (S start, E end, Q quit)
python -m gazedemo cue                                      # 4. accuracy test (15 cues)
python -m gazedemo calibrate --run runs/<date>_cue          #    calibrate p_look from that cue run
python -m gazedemo collect --participant P01 --trials 30    # 5. collect food-choice data
python -m gazedemo dataset --runs runs --out data/gaze_choice.csv   # 6. build the dataset
python -m gazedemo train --data data/gaze_choice.csv --out models  #    train + evaluate the classifier
python -m gazedemo report --out report                      #    one-page summary → report/SUMMARY.md
python -m gazedemo live --model models/model.json           #    live view with p_choice
```
Replace `runs/<date>_cue` with the folder name the `cue` command printed, for example `runs/20261029_110028_cue`.

### Step 0 — Check the machine
```bash
python -m gazedemo doctor
```
The last line should read `CUDA OK · camera N OK · Gazelle OK`. `doctor` lists every camera index it can open, with resolution and fps.
- **Camera not OK:** use the index it suggests, e.g. `--set source.device=4`.
- **RealSense D435i:** it shows up as several indices (colour, infrared, depth); pick the colour one at 1280×720. It also needs `source.fourcc: null`. A normal webcam on Windows works best with `MJPG`.
- **Make it permanent:** edit `source.device` / `source.fourcc` in `configs/default.yaml` so you don't need the flag every time.

### Step 1 — Measure speed (optional)
```bash
python -m gazedemo bench --n 200
```
It writes `runs/<date>_bench/bench.md` with timings for ViT-L / ViT-B × fp32 / fp16. If ViT-L is under about 5 Hz, set `gazelle.name: gazelle_dinov2_vitb14_inout` in `configs/default.yaml`.

### Step 2 — Draw the three plates
Set up the camera and plates in their final positions, then:
```bash
python -m gazedemo setup-aoi
```
| Action | Effect |
| --- | --- |
| Drag with the left mouse button | Draw a box around plate A, then B, then C (an ellipse is fitted inside) |
| Click or right-click | Mark the person's head (to check the angles between plates) |
| `+` / `-` | Enlarge / shrink the ellipses (margin) |
| `U` | Undo |
| `R` | Take a new picture from the camera |
| `Enter` | Save `aoi.json` (refused if ellipses overlap) |
| `Q` | Quit without saving |

Redo this whenever the camera or the plates move.

### Step 3 — Live check
```bash
python -m gazedemo live
```
| Key | Effect |
| --- | --- |
| `S` | Start a round |
| `E` | End the round |
| `1` / `2` / `3` | Label the round with plate A / B / C (where you really looked) |
| `H` | Show / hide the heatmap |
| `Q` | Quit |

Click on the window first so it receives the keys. Quick test: press S, look at plate A for 3 s, then B for 2 s, press E. The table should show DD ≈ 3000 ms for A and ≈ 2000 ms for B.

### Step 4 — Measure "which plate" accuracy
```bash
python -m gazedemo cue
```
1. Press **Space**.
2. The screen shows "LOOK AT A/B/C" with a beep, 15 times in random order (5 per plate).
3. Look at the named plate until the next beep.

At the end it prints the accuracy and confusion matrix (also saved as `score.md`). Then calibrate p_look from this run:
```bash
python -m gazedemo calibrate --run runs/<date>_cue
```

### Step 5 — Collect food-choice data
```bash
python -m gazedemo collect --participant P01 --trials 30
```
1. Rate each food from 1 (not at all) to 5 (very much).
2. For each round, place the food cards as shown ("A = rice · B = egg · C = vegetables") and press **Space**.
3. After the beep, look at the plates and decide which food you want. A second beep ends the round, 5 s later.
4. Press **1 / 2 / 3** for the plate you chose, or **X** to skip the round.

Other keys: **P** pauses (the current round is redone) and **Q** saves and quits; finished rounds are kept. The plate positions of the foods are balanced automatically across rounds.

### Step 6 — Train and report
```bash
python -m gazedemo dataset --runs runs --out data/gaze_choice.csv
python -m gazedemo train --data data/gaze_choice.csv --out models
python -m gazedemo report --out report
python -m gazedemo live --model models/model.json     # live view now also shows p_choice
```

**Tips:**
- **Change a setting for one run** with `--set key=value`, e.g. `--set gazelle.name=gazelle_dinov2_vitb14_inout`.
- **Try it without a camera or GPU:** `python -m gazedemo live --source synthetic --fake-gaze`.
- **See every command:** `python -m gazedemo --help`.

---

## 4. How to use the results after running

### Session folders
Every live / cue / collect session creates `runs/<YYYYmmdd_HHMMSS>_<command>/`:

| File | Content |
| --- | --- |
| `trials.csv` | **One row per round**: for each plate `A_ttfd_ms`, `A_dd_ms`, `A_dc`, `A_p_look`, … plus the label/choice, foods and liking (collect), and data quality (`valid_ratio`, `eff_hz`). Opens in Excel. |
| `dwells.csv` | Every dwell: plate, start, end, duration |
| `samples.jsonl` | Every processed frame: label, heatmap share per plate, gaze point, in/out score, latency |
| `events.jsonl` | Every event with its time: round start/end, keys, labels, beeps |
| `video.mp4` + `frames.csv` | The recorded video with capture times (for `offline`) |
| `snapshot.png` | A screenshot of the live window |
| `config.yaml`, `aoi.json`, `run.json` | The exact settings, plate ellipses and session info used |
| `session.json`, `schedule.csv` | (collect) liking scores and the food layout of each round |
| `score.md` | (cue) accuracy and confusion matrix |

### Reading the numbers
- **"Looked at" vs "chosen":** the plate with the **largest DD** is the plate looked at most; that's how `score` judges "which plate". **p_look** answers "where is the person looking"; **p_choice** answers "which food will they choose". They are different questions.
- **A dwell is a look of at least 100 ms.** Short glances and gaps under 150 ms are merged, and `configs/default.yaml` holds all the thresholds.
- **`valid_ratio` near 1.0 is good.** Low values mean the camera or Gazelle had problems (rounds below 0.6 are dropped from the dataset).

### Follow-up commands
| Command | Use |
| --- | --- |
| `python -m gazedemo score --run runs/<id>` | Accuracy / confusion / reliability of any run that has labels (writes `score.md`) |
| `python -m gazedemo calibrate --run runs/<cue run>` | Writes `models/calibration.json`; later sessions add a calibrated `p_look_cal` column |
| `python -m gazedemo offline --run runs/<id>` | Re-processes the recorded video with every frame (or with new settings/model) → `runs/<id>/offline/` (`--stride 2` = every 2nd frame) |
| `python -m gazedemo dataset ...` | Builds `data/gaze_choice.csv` (one row per round × plate) from all collect runs; the offline results are used when present |
| `python -m gazedemo train ...` | Compares baselines with the conditional-logit classifier (cross-validated) → `models/report.md`, `models/eval.json`, `models/model.json` |
| `python -m gazedemo report ...` | `report/SUMMARY.md`: speed, cue accuracy, collected data, classifier table, an overlay picture and a timeline, all in one page |
| `python -m gazedemo live --model models/model.json` | Shows p_choice live (also works with `collect` and `offline`) |

### Backups
`video.mp4` files are large (about 70 MB per minute), so `.gitignore` keeps them out of git. Commit the rest with `git add -A; git commit -m "..."; git push`, and copy `runs/` to a USB stick or Drive to keep the videos.
