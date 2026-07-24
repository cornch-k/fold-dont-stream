# rdepth — 순환 깊이 30M 프로토타입
설계: ../../docs/specs/2026-07-23-포스트트랜스포머-트랙-설계.md
계획: ../../docs/plans/2026-07-23-rdepth-prototype.md

## 캘리브레이션 확정 (2026-07-23)
- dtype: fp32 (스모크: fp32 69.7k > fp16 60.8k > bf16 44.5k tok/s; fp16은 스케일러 없이 수렴 손상)
- micro-batch: 64 (large 21.7k tok/s), 유효 배치 32768 tok/step
- 토큰 예산: 200,000,000/런 (6,103스텝), 3런 총 ~7시간
- 데이터: train 480.7M tok / val 4.84M tok (BPE-4096)
