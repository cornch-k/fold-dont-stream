# Night Results — 2026-07-23/24 야간 측정

작업 디렉터리: `/Users/angigyeom/Desktop/optillama/llama.cpp`
안전 규칙: llama 프로세스 동시 1개, Metal 금지(-ngl 0), llama-bench 금지, 워치독 패턴(watchdog: `[ Prompt:` 등장 / 스왑델타>1500MB / 제한시간 초과 시 kill -9)
공통 env: `LLAMA_MMAP_NO_PREFETCH=1 -t 8 -fa on --no-warmup -no-cnv --no-repack < /dev/null`

상태: **완료** (T1-T4 전체 성공, 실패 런 없음) — 종료 시각 01:29 KST, llama 프로세스 잔존 없음 확인, 디스크 여유 375GB

## T1. IQ2_XXS 다운로드

- 소스: `bartowski/Llama-3.3-70B-Instruct-GGUF` (HF API 200 확인, bartowski 정상 존재하여 unsloth 폴백 불필요)
- 명령: `HF_HUB_DISABLE_XET=1 hf download bartowski/Llama-3.3-70B-Instruct-GGUF --include "*IQ2_XXS*" --local-dir .../models/Llama-3.3-70B-IQ2/`
- 결과 파일: `/Users/angigyeom/Desktop/optillama/models/Llama-3.3-70B-IQ2/Llama-3.3-70B-Instruct-IQ2_XXS.gguf`
- 크기: 19,097,390,048 bytes (≈17.8GiB, 단일 파일 — 샤딩 없음)
- 소요 시간: 약 00:31:xx ~ 00:42:23 (약 11분), 상태: 성공
- 다운로드 중 측정 없음 (규칙 준수)

## T2. 양자화 베이스라인 런 (IQ2_XXS, -n 32)

모델: `/Users/angigyeom/Desktop/optillama/models/Llama-3.3-70B-IQ2/Llama-3.3-70B-Instruct-IQ2_XXS.gguf` (2.0625 bpw)
명령: `LLAMA_MMAP_NO_PREFETCH=1 llama-cli -m <IQ2_XXS> -p "The story begins" -n 32 -t 8 -fa on -ngl 0 --no-warmup -no-cnv --no-repack < /dev/null` (워치독, 제한 1800초)

| 런 | 상태 | Prompt t/s | Generation t/s | elapsed | peakRSS | ssd_read | swap_delta | 비고 |
|---|---|---|---|---|---|---|---|---|
| t2_iq2xxs | OK | 1.1 | 1.7 | 98s | 20GB | 30GB | -8MB | `[ Prompt:` 마커 검출로 정상 종료. 이후 stdin=/dev/null EOF로 인한 빈 `> ` 루프 발생했으나 워치독이 이미 kill 완료(로그에 지장 없음) |

생성 텍스트 (연속 문단, 문장 단위로 표기 — n=32 토큰 제한으로 마지막 문장 중간 절단):
```
The story begins...in a small, seaside town where the misty dawn air clung to the streets like a damp,
grey shroud. It was a place where time
```
Coherence 판정: 문법적으로 온전한 영어 문장 생성 확인 (IQ2_XXS치고 양호 — "seaside town", "misty dawn air clung to the streets like a damp, grey shroud" 등 자연스러운 구문). 붕괴/반복/난센스 없음 → PASS.

## T3. 접기 반복 측정 (분산용, 각 3회)

### llama_fold (LLAMA_FOLD_K=40, Llama-3.3-70B Q4_K_M, -n 32, 제한 900초)

| 회차 | 시각(KST) | 상태 | elapsed | Generation t/s | Prompt t/s | peakRSS | ssd_read | swap_delta |
|---|---|---|---|---|---|---|---|---|
| 1 | 00:46 | OK | 92s | 2.5 | 1.1 | 21GB | 33GB | 0MB |
| 2 | 00:48 | OK | 93s | 2.3 | 1.1 | 21GB | 31GB | -16MB |
| 3 | 00:50 | OK | 93s | 2.4 | 1.0 | 21GB | 31GB | 0MB |

평균 ± 표준편차 (Generation t/s): **2.40 ± 0.10** t/s (n=3, 표본표준편차)

### laguna_fold (LLAMA_FOLD_K=16, Laguna-S-2.1 UD-IQ4_XS, -n 32, 제한 900초)

| 회차 | 시각(KST) | 상태 | elapsed | Generation t/s | Prompt t/s | peakRSS | ssd_read | swap_delta |
|---|---|---|---|---|---|---|---|---|
| 1 | 00:52 | OK | 78s | 1.9 | 1.7 | 20GB | 18GB | -8MB |
| 2 | 00:54 | OK | 78s | 2.7 | 1.8 | 15GB | 19GB | -8MB |
| 3 | 00:55 | OK | 72s | 2.5 | 1.8 | 15GB | 20GB | 0MB |

평균 ± 표준편차 (Generation t/s): **2.37 ± 0.42** t/s (n=3, 표본표준편차) — llama_fold보다 분산이 큼(peakRSS도 20→15GB로 회차 간 변동)

## T4. 토큰 수 통일 기준선 (FOLD 미설정, Q4_K_M 70B, -n 32, 제한 5400초)

명령: `LLAMA_MMAP_NO_PREFETCH=1 llama-cli -m <Q4_K_M 70B> -p "The story begins" -n 32 -t 8 -fa on -ngl 0 --no-warmup -no-cnv --no-repack < /dev/null` (워치독, 제한 5400초, FOLD 환경변수 미설정)

| 런 | 상태 | elapsed | Prompt t/s | Generation t/s | peakRSS | ssd_read | swap_delta | 비고 |
|---|---|---|---|---|---|---|---|---|
| t4_streaming_base | OK | 1713s (~28.5분) | 0.6 | 0.0* | 21GB | **1298GB** | -165MB | *반올림으로 0.0 표시 (실제로는 0.05 t/s 미만 추정) — n=32 전체가 완료되어 `[ Prompt:` 마커로 정상 종료 |

생성 텍스트 (연속 문단, n=32 토큰 제한으로 마지막 문장 중간 절단):
```
The story begins...on a stormy night, in a small, coastal town, where the rain poured down like a
relentless curtain of grey silk. The streets were empty,
```

**해석**: 접기(FOLD) 없이 42GB Q4_K_M 70B를 스트리밍하면 32토큰 생성에 1298GB의 SSD 재읽기가 발생 — 모델 크기(42GB)의 약 31배. T3의 llama_fold(K=40) 대비(ssd_read 31~33GB, 동일 -n 32) 약 40배의 디스크 트래픽이며, Generation 처리량은 2.4 t/s → 사실상 0 t/s로 붕괴. 이는 "절벽"(FOLD 미적용 시 토큰마다 콜드 페이지를 반복적으로 재로드하는 현상)이 실측으로 재확인된 것 — 접기 최적화가 디스크 I/O를 극적으로 줄여준다는 가설을 강하게 지지.

## 실패 로그

없음 — T1~T4 전체 10개 런(다운로드 1 + T2 1 + T3 6 + T4 1) 모두 정상 완료(st=OK). 안전 규칙 위반(Metal 사용, 동시 실행, 다운로드 중 측정) 없음.

비고: 각 llama-cli 실행 후 `[ Prompt:` 성능 마커 검출 시 워치독이 즉시 kill -9 하는데, 이때 stdin이 `/dev/null`이라 종료 전 짧게 빈 `> ` 프롬프트 루프가 로그에 반복 기록되는 현상이 T2/T4 로그에서 관찰됨(정상 동작, 측정치에는 영향 없음 — perf 마커는 이미 기록된 후의 잔여 출력).

## 원본 로그 경로

- T1: `/tmp/night_ops/t1_download.log`
- T2: `/tmp/night_ops/t2_iq2xxs.log`
- T3 llama_fold: `/tmp/night_ops/t3_llamafold_r1.log`, `..._r2.log`, `..._r3.log`
- T3 laguna_fold: `/tmp/night_ops/t3_lagunafold_r1.log`, `..._r2.log`, `..._r3.log`
- T4: `/tmp/night_ops/t4_streaming_base.log` (wrapper: `/tmp/night_ops/t4_wrapper.log`)
- 워치독 스크립트: `/Users/angigyeom/Desktop/optillama/protolab/night_watchdog.sh`

## 야간 종합

**T2 — IQ2_XXS 양자화 베이스라인** (2.0625bpw, -ngl 0, -n 32): OK, Prompt 1.1 t/s / **Generation 1.7 t/s**, elapsed 98s, peakRSS 20GB, ssd_read 30GB, swap_delta -8MB. 생성문: *"...in a small, seaside town where the misty dawn air clung to the streets like a damp, grey shroud. It was a place where time"* — 문법·의미 모두 온전한 영어 산문, 반복/붕괴/난센스 없음 → **Coherence PASS** (IQ2치고 양호, 극단적 양자화에도 문장 구조 유지 확인).

**T3 — 접기(FOLD) 반복 측정 (각 n=3)**:
- llama_fold (LLAMA_FOLD_K=40, Q4_K_M 70B): **2.40 ± 0.10 t/s** (분산 작음, peakRSS 21GB로 3회 모두 일관)
- laguna_fold (LLAMA_FOLD_K=16, Laguna-S-2.1 UD-IQ4_XS): **2.37 ± 0.42 t/s** (분산 큼, peakRSS 20→15→15GB로 변동 — 캐시 상태에 따른 회차 간 편차 시사)

**T4 — 접기 없는 스트리밍 기준선** (Q4_K_M 70B, -n 32): OK, Prompt 0.6 t/s / **Generation ≈0.0 t/s**(반올림, 실제 <0.05 t/s 추정), elapsed 1713s(~28.5분), peakRSS 21GB, **ssd_read 1298GB**, swap_delta -165MB.

**3자 비교 (접기 vs IQ2 양자화 vs 무접기 스트리밍)**: 동일 -n 32, 동일 70B급 모델 기준으로 세 접근이 극명하게 갈린다. 접기(FOLD)는 ssd_read를 31~33GB(모델 크기 수준)로 억제해 2.4 t/s를 내는 반면, 무접기 스트리밍은 정확히 같은 모델·같은 토큰 수에서 ssd_read가 1298GB(약 40배)까지 폭증하며 Generation이 사실상 0으로 붕괴한다 — 이는 폴딩 없이는 토큰마다 콜드 페이지를 반복 재로드하는 "메모리 절벽"이 실측으로 재현된 것이다. IQ2_XXS 양자화(파일 크기를 42GB→17.8GB로 줄이는 접근)는 접기 없이도 1.7 t/s를 내지만 이는 모델 자체가 작아져 페이지 폴트 총량이 줄었기 때문이며, 접기(2.4 t/s)에는 못 미친다. 종합하면 이번 야간 배치에서는 **접기(FOLD) > IQ2 양자화 >> 무접기 스트리밍** 순으로, 접기가 디스크 I/O 절감 및 처리량 양면에서 가장 효과적인 완화책임을 재확인했고, 양자화는 차선책, 무접기 스트리밍은 70B급에서 사실상 사용 불가능한 절벽임이 정량적으로 재입증되었다.
