# TPU 스케일링 사다리 — Phase 1 (dense)

`training/rdepth/`의 15M TinyStories 연구를 JAX/Flax로 옮겨 15M → 50M → 150M
회복률 추세선을 뽑는다. TRC 신청서 Stage 1의 축소 리허설이고, 여기서 나온
기울기가 "41%가 스케일에서 살아남는가"에 대한 첫 실측치다.

```
recovery = (val_small − val_loop) / (val_small − val_large)
```

2× 메모리로 얻는 품질 격차 중 looping이 되찾는 비율. PyTorch 15M 참조값은
`1.5950 / 1.5577 / 1.5038` → **0.4090**.

**Phase 2(MoE)는 아직 없다.** 15M 게이트를 통과하기 전에는 쓰지 않는다.

## 파일

| 파일 | 역할 |
|---|---|
| `model.py` | 순환 깊이 GPT (dense). `training/rdepth/model.py`의 dense 경로 포트 |
| `train.py` | 학습 루프, 배치 축 샤딩, 체크포인트/재개, CSV + JSON |
| `prepare_fineweb.py` | FineWeb-Edu → 4k BPE → uint16 bin |
| `scaling.py` | 회복률 표 + 게이트 판정 + 추세선 PNG |
| `test_model.py` | `python test_model.py` — CPU 몇 초. TPU 태우기 전 필수 |
| `rdepth_tpu_colab.ipynb` | Colab TPU v6e-1 실행 노트북 |

## 코퍼스: 게이트는 TinyStories, 사다리는 FineWeb-Edu

TinyStories로 사다리 전체를 돌릴 수 없다. `train.bin`이 4k BPE 기준 480.7M
토큰인데 13:1 비율이 요구하는 양은:

| 스케일 | 필요 토큰 | TinyStories 에폭 |
|---|---|---|
| 15m | 197.7M | 0.41 |
| 50m | 644.7M | **1.34** |
| 150m | 1.864B | **3.87** |

50m부터 데이터가 없고 150m은 4바퀴 반복이다. 반복 학습에서는 large arm이 더
많이 암기하므로 `small − large` 격차가 깊이와 무관한 이유로 벌어져 회복률이
낮게 나온다. 반대 방향으로는, TinyStories가 3~4세 어휘 ~1500단어로 제한된
합성 코퍼스라 용량을 키우면 두 arm이 같은 엔트로피 바닥으로 수렴해 격차가
줄어든다 (15m에서 이미 0.0912 nats). 두 오염이 방향이 반대라 상쇄도 기대할 수
없다. `train.py`는 `train.bin`이 요구 토큰보다 짧으면 assert로 죽는다.

그런데 게이트를 FineWeb-Edu에서 돌리면 회복률이 0.33으로 나왔을 때 **JAX 포트
버그인지 코퍼스 효과인지 구분이 안 된다.** 0.41±0.04는 TinyStories 숫자다.
그래서:

- **게이트 (15m)** — TinyStories, 197.7M 토큰(0.41 에폭). PyTorch 런과 배치·LR·
  워밍업·시드까지 동일. 회복률이 0.41±0.04를 벗어나면 포트 문제로 보고 멈춘다.
- **사다리 (15m/50m/150m)** — FineWeb-Edu. 15m이 두 코퍼스에 다 있으므로
  코퍼스 효과 자체도 이 지점에서 읽힌다.

`scaling.py`는 `--data-dir` 이름(`tinystories*`)으로 게이트 런을 구분한다.

## 사다리

구조는 `1 pre / 2 loop / 1 post`, `ctx=512`, `vocab=4096`, tied embedding으로
고정하고 `d`만 키운다. 회복률을 스케일 축 하나에서만 움직이게 하려는 것.

| 스케일 | d | heads | ffn | small | loop (R=3) | large (2× depth) | 토큰 (13:1) | 스텝 |
|---|---|---|---|---|---|---|---|---|
| 15m | 512 | 8 | 1408 | 15,208,960 | 15,210,496 | 28,058,112 | 197,716,480 | 6,033 |
| 50m | 960 | 15 | 2640 | 49,590,720 | 49,593,600 | 94,756,800 | 644,679,360 | 19,674 |
| 150m | 1664 | 26 | 4576 | 143,358,592 | 143,363,584 | 279,047,808 | 1,863,661,696 | 56,874 |

- loop arm이 small보다 큰 만큼(`R·d`)은 루프 임베딩뿐이다. 15m에서 1,536개,
  0.01%. `test_model.py`가 이 차이가 정확히 `R·d`인지 검사한다 — 더 크면
  middle-cycle 공유가 깨진 것이다.
- 토큰 예산은 **스케일당 고정**(small arm 파라미터 × 13). large가 크다고 토큰을
  늘리지 않는다. 세 arm이 같은 토큰을 봐야 회복률이 성립한다.
- vocab 4096을 유지하는 이유: GPT-2 어휘(50257)면 d=512에서 임베딩만 25.7M
  파라미터라 15m 칸 자체가 성립하지 않는다. 대신 FineWeb-Edu에 4k BPE를 새로
  학습시킨다. 13:1의 "토큰"은 전부 이 토큰 단위다.
- LR 6e-4는 세 스케일 공통. arm 비교는 스케일 안에서만 하므로 LR이 150m에
  최적이 아니어도 회복률을 편향시키지 않는다. 대신 스케일 간 절대 loss는
  Chinchilla-최적 런과 비교하면 안 된다.

## 실행

```sh
python test_model.py                                  # 먼저. 몇 초
python ../rdepth/prepare_data.py                      # TinyStories (게이트용)
python prepare_fineweb.py --out data/fineweb          # 1.9B 토큰, train.bin ≈ 3.8 GB

# 게이트
for arm in small loop large; do
  python train.py --run 15m-$arm --data-dir data/tinystories --resume
done
python scaling.py                                     # 0.41±0.04 판정

# 통과했을 때만 사다리
for s in 15m 50m 150m; do for arm in small loop large; do
  python train.py --run $s-$arm --data-dir data/fineweb --resume
done; done
python scaling.py --plot results/recovery.png
```

`--resume`는 항상 붙인다. 세션이 끊겨도 같은 명령을 다시 치면 이어간다
(배치 샘플링이 `[seed, step]`으로 stateless라 재개 후에도 같은 배치가 나온다).

## CU 예산

Phase 1 dense 전체 = **8.54e18 FLOPs** (6ND 기준).

| 스케일 | FLOPs | 비중 |
|---|---|---|
| 15m | 7.6e16 | 1% |
| 50m | 8.7e17 | 10% |
| **150m** | **7.6e18** | **89%** |

v6e-1 peak bf16 918 TFLOP/s 기준 소요 시간:

| MFU | 시간 |
|---|---|
| 40% | 6.5 h |
| 25% | 10.3 h |
| 15% | 17.2 h |

`ctx=512`, 배치 32,768 토큰은 v6e에 작은 스텝이라 MFU가 15~25%에 머물 가능성이
높다. 프로토콜상 배치를 못 키우므로 이건 감수해야 한다. **첫 런 전에
`--max-seconds 600`으로 실측 tok/s를 뽑아 위 표를 실제 숫자로 갈아끼울 것.**

150m이 89%를 먹는다. CU가 모자라면 잘라낼 곳은 여기다 — 15m+50m만 해도
11%(대략 1~2시간)로 추세선의 두 점은 확보된다.

체크포인트 크기: 150m-large는 파라미터 279M × 4B + Adam 2슬롯 = **3.3 GB**.
Drive 무료 15GB에 데이터 3.8GB까지 얹으면 빠듯하다. 200스텝마다 3.3GB를 Drive에
쓰면 학습보다 쓰기가 오래 걸리므로 `--ckpt-every`를 스케일별로 올린다
(15m 200 / 50m 500 / 150m 2000). 끝난 런의 체크포인트는 바로 지운다.

## Phase 2 (아직 없음)

게이트 통과 후 추가한다. 확정된 요구사항:

1. aux loss 누적 횟수를 sticky와 reroute에서 **동일하게** 맞춘다. sticky가
   라우팅 결정을 재사용해도 load-balance 통계는 두 패스 모두에서 누적.
   (`training/rdepth/model.py`는 sticky 1회 / reroute 2회로 달라 confound였다.)
2. sticky = 첫 패스의 expert 선택 재사용, 라우터 재계산 없음.
3. expert-touch: (token, layer-slot)당 루프 누적 unique expert 수. looped block과
   unshared block을 분리 기록.
4. XLA 정적 shape. capacity factor 디스패치. 드롭 토큰 비율을 매 스텝 로그.
5. 시드 3개(1337/1338/1339). sticky−reroute 차이가 시드 노이즈 수준이라 필수.
