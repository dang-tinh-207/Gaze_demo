# Bench — 2026-10-29T00:25:14

- Máy: Linux-6.8.0-138-generic-x86_64-with-glibc2.35 · Python 3.10.22 · torch 2.5.1+cu118 · GPU NVIDIA GeForce RTX 4060 Laptop GPU
- Nguồn frame: synthetic (1280×720, không phải camera → bỏ qua thời gian đọc)
- n = 200 frame mỗi cấu hình. Gazelle = BGR→RGB + tiền xử lý + model + về CPU; gán đĩa = assign_label + argmax. Hz cuối = min(Hz xử lý, fps camera).

| Model | dtype | Gazelle mean (ms) | Gazelle p95 (ms) | Gán đĩa mean (ms) | Gán đĩa p95 (ms) | Hz xử lý | Hz cuối |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-L (`gazelle_dinov2_vitl14_inout`) | fp32 | 170.8 | 174.1 | 0.11 | 0.13 | 5.8 | 5.8 |
| ViT-L (`gazelle_dinov2_vitl14_inout`) | fp16 | 57.9 | 58.9 | 0.10 | 0.11 | 17.2 | 17.2 |
| ViT-B (`gazelle_dinov2_vitb14_inout`) | fp32 | 60.6 | 63.5 | 0.11 | 0.12 | 16.5 | 16.5 |
| ViT-B (`gazelle_dinov2_vitb14_inout`) | fp16 | 23.6 | 24.2 | 0.10 | 0.11 | 42.2 | 42.2 |

**Khuyến nghị:** ViT-L fp16 đạt 17.2 Hz (≥ 5) → giữ gazelle.name = gazelle_dinov2_vitl14_inout, gazelle.fp16 = True.
Độ phân giải thời gian của DD/TTFD ≈ 1/Hz.
