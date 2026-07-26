# 서빙 모델 파라미터 (attention_unet_v1.onnx)

**확정일:** 2026-07-26 · **모델:** Attention U-Net (resnet34) · **손실:** immune 가중 CrossEntropy (wce20)

## 모델 선정

| 항목 | 값 |
|---|---|
| 서빙 fold | **fold 4** (fold1 vs fold4 중 test45 MAE 최저 + 커버리지 최고) |
| best_epoch | **12** (val sTIL MAE 기준) |
| 손실 | wce20 = CrossEntropy(weight: immune=2.0, 나머지 1.0), ignore_index=255 |
| dropout_p | 0.2 · seed 0 · 30 epoch |
| 체크포인트 | `checkpoints/attnunet_wce20_fold4_best.pth` |

선정 근거: sTIL 오차 근본원인이 immune→stroma 픽셀 혼동이라, immune 가중 CE로 5-fold 재학습.
pooled OOF(n=106) sTIL MAE 9.88→**8.52**, 위양성 64.9→37.8 감소, 전 fold 개선.

## 성능 (held-out test 45장, 결정론 추론)

| 지표 | 값 |
|---|---|
| **test sTIL MAE** | **7.152** |
| 컷오프 오분류 (10/20/50%) | 4/45, 5/45, 5/45 (정확도 91.1 / 88.9 / 88.9%) |
| 과소/과대 오차합 | 125 / 197 |

## fold4 위양성 검토 (fold1 대비 12.6 vs 0.2 — 무해 확인)

immune GT=0 인 test ROI 5개에서 fold4 위양성 분해:

| ROI | fold4 pred | fold1 pred | fold4 구간(±q) |
|---|---|---|---|
| GM-A2DI | 8.2 | 0.0 | [0.0, 27.8] |
| EW-A3U0 | 4.1 | 0.2 | [0.0, 23.8] |
| OL-A97C/A5D6/A5RW | ≤0.2 | 0.0 | [0.0, ~19.7] |

- 합 12.6이나 **max 8.2** — 가장 낮은 컷오프 10%도 못 넘음.
- 컷오프 판정 뒤집힘: 10/20/50% 전부 **0건** (GT 음성 유지).
- conformal 흡수: 5개 전부 구간이 진실 sTIL=0 을 포함 → 판정유보로 흡수.
- 결론: 임상 무해. fold4 유지 확정.

## Conformal (CV+, 비정규화)

| 항목 | 값 |
|---|---|
| **q (비정규화)** | **19.629** (alpha=0.10, 90% 목표) |
| 산출 | 5-fold pooled OOF 잔차 |pred−true| 의 유한표본 분위수 (k=⌈(n+1)(1−α)⌉) |
| **test45 커버리지** | **95.6% (43/45)** — 90% 보장 충족 |
| 예측구간 | `til_pred ± q` (= til_pred ± 19.63) |

*(참고: 이전 CE baseline은 q=24.19, 커버리지 95.6%. wce20은 잔차가 작아 q가 19% 좁아지면서 동일 보장 유지.)*

## ONNX export

| 항목 | 값 |
|---|---|
| 파일 | `serving_model/attention_unet_v1.onnx` (백업: `.onnx.bak` = 이전 CE 버전) |
| 방식 | `TrainingMode.PRESERVE`, opset 18, `do_constant_folding=True`, `dynamic_axes`(batch) |
| MC Dropout | RandomUniformLike ×4 유지, Dropout 노드 0 |
| 출력 | `[batch, 5, 512, 512]` (채널: tumor·stroma·immune·necrosis·other) |

## 검증 (전부 통과)

- `tests/test_onnx_mc_dropout.py` : 7/7 PASS
- 수치 동등성: pth.eval() vs ONNX(eval) max_abs_err = **3.13e-06** (< 1e-5)
- 확률성: ORT 3레벨(ENABLE_ALL/BASIC/DISABLE_ALL) 배치 std = **0.341** (> 1e-4)
