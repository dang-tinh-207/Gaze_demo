# Bench — 2026-10-29T10:07:36

- Máy: Linux-6.8.0-138-generic-x86_64-with-glibc2.35 · Python 3.10.22 · torch 2.5.1+cu118 · GPU NVIDIA GeForce RTX 4060 Laptop GPU
- Nguồn frame: camera 4 (1280×720, mặc định)
- Đọc frame: mean 69.5 ms, p95 69.2 ms (≈ 15.0 fps camera)
- n = 200 frame mỗi cấu hình. Gazelle = BGR→RGB + tiền xử lý + model + về CPU; gán đĩa = assign_label + argmax. Hz cuối = min(Hz xử lý, fps camera).

| Model | dtype | Gazelle mean (ms) | Gazelle p95 (ms) | Gán đĩa mean (ms) | Gán đĩa p95 (ms) | Hz xử lý | Hz cuối |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-L (`gazelle_dinov2_vitl14_inout`) | fp32 | 171.6 | 175.2 | 0.12 | 0.13 | 5.8 | 5.8 |
| ViT-L (`gazelle_dinov2_vitl14_inout`) | fp16 | 58.1 | 59.1 | 0.11 | 0.13 | 17.2 | 15.0 |
| ViT-B (`gazelle_dinov2_vitb14_inout`) | fp32 | 60.2 | 63.3 | 0.11 | 0.12 | 16.6 | 15.0 |
| ViT-B (`gazelle_dinov2_vitb14_inout`) | fp16 | 23.7 | 24.2 | 0.10 | 0.11 | 42.1 | 15.0 |

**Khuyến nghị:** ViT-L fp16 đạt 15.0 Hz (≥ 5) → giữ gazelle.name = gazelle_dinov2_vitl14_inout, gazelle.fp16 = True.
Độ phân giải thời gian của DD/TTFD ≈ 1/Hz.
