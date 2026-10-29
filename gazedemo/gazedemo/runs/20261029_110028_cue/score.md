# Score — 20261029_110028_cue

- Trial có nhãn: 15 (A 5 · B 5 · C 5)
- **Accuracy (đĩa có DD lớn nhất = nhãn): 0/15 = 0.00**
- Accuracy theo argmax p_look: 6/15 = 0.40
- Tỉ lệ thời gian hợp lệ trung bình (valid_ratio): 1.00

## Confusion — dự đoán = đĩa có DD lớn nhất (tất cả 0 → none)
| thật \ dự đoán | A | B | C | none |
| --- | --- | --- | --- | --- |
| A | 0 | 1 | 0 | 4 |
| B | 1 | 0 | 0 | 4 |
| C | 0 | 1 | 0 | 4 |

## Confusion — dự đoán = argmax p_look
| thật \ dự đoán | A | B | C | none |
| --- | --- | --- | --- | --- |
| A | 5 | 0 | 0 | 0 |
| B | 4 | 1 | 0 | 0 |
| C | 2 | 3 | 0 | 0 |

## Reliability p_look theo frame (612 frame hợp lệ trong các trial)
q_max = max của q trên A/B/C/none; đúng = argmax q là đĩa có nhãn.

| bin q_max | n frame | q_max TB | tỉ lệ đúng |
| --- | --- | --- | --- |
| 0.0–0.2 | 0 | — | — |
| 0.2–0.4 | 0 | — | — |
| 0.4–0.6 | 10 | 0.55 | 0.00 |
| 0.6–0.8 | 20 | 0.70 | 0.00 |
| 0.8–1.0 | 582 | 0.95 | 0.00 |

ECE = 0.938
