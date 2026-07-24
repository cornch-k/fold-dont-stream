# 펼쳐진 쌍둥이(flattened twin) 실행 프로토콜 — Nanbeige4.2-3B

작성: 코드 준비 담당 (오늘 밤은 스크립트/소스 검증만, 실측/변환은 금지)
관련 파일: `flatten_gguf.py`, `ballast.py`, `nanbeige-llamacpp/src/models/nanbeige.cpp`

## 0. V1 핵심 발견 (실행 전 반드시 확인)

Nanbeige의 루프는 **순수 반복이 아닙니다.** `llama_model_nanbeige::graph()`
(`nanbeige-llamacpp/src/models/nanbeige.cpp:160-167`)는 루프 경계마다
`output_norm` 가중치를 재사용하는 RMSNorm을 한 번 더 끼워 넣습니다:

```cpp
if (n_loops > 1 &&
    ((il + 1) % n_phys) == 0 &&
    (il + 1) < n_layer &&
    !nb.skip_loop_final_norm) {
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "loop_norm", il);
    inpL = cur;
}
```

이 삽입은 `n_loops > 1`에만 걸려 있습니다. 실제 배포된
`nanbeige4.2-3b-Q4_K_M.gguf`를 직접 읽어 확인한 값:

```
nanbeige.block_count            = 22
nanbeige.num_loops               = 2
nanbeige.skip_loop_final_norm    = False
```

`skip_loop_final_norm=False`이므로 이 boundary norm은 **실제로 실행됩니다.**
RMSNorm 두 번을 대수적으로 합칠 수도 없습니다 (`RMSNorm(RMSNorm(x,w1),w2) ≠
RMSNorm(x,w1·w2)` — 중간에 rms(·) 재정규화가 끼기 때문에 잔차 경로 자체가
달라짐). 따라서 `block_count=44, num_loops=1`로만 메타데이터를 바꾸는 순수
펼치기는 **원본과 수학적으로 동일하지 않습니다** — `n_loops>1` 게이트가
꺼지는 순간 이 norm이 통째로 사라집니다.

**필요한 조정 (소스 패치, 오늘 밤은 적용하지 않음):**

`nanbeige.cpp`에 새 필드/KV를 하나 추가해 `n_loops==1`이어도 원래의 물리
스트라이드(22)에서 boundary norm을 재삽입하도록 게이트를 분리해야 합니다.
`flatten_gguf.py`가 이미 `{arch}.flatten.source_block_count` KV를 출력
파일에 써 두므로, 이를 그대로 재사용할 수 있습니다:

```cpp
// models.h: layers 433-435 부근에 필드 추가
int unroll_boundary_stride = 0;   // 0 = 비활성

// nanbeige.cpp load_arch_hparams()
uint32_t stride_u = 0;
ml.get_key(LLM_KV_..._FLATTEN_SOURCE_BLOCK_COUNT /* 또는 문자열 키 직접 조회 */,
           stride_u, false);
unroll_boundary_stride = (int) stride_u;

// nanbeige.cpp graph(), 경계 조건 교체
const int boundary_stride  = nb.unroll_boundary_stride > 0 ? nb.unroll_boundary_stride : n_phys;
const bool has_boundary    = (n_loops > 1) || (nb.unroll_boundary_stride > 0);
if (has_boundary &&
    ((il + 1) % boundary_stride) == 0 &&
    (il + 1) < n_layer &&
    !nb.skip_loop_final_norm) {
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, il);
    inpL = cur;
}
```

`(il+1) < n_layer` 조건 덕분에 마지막 물리 레이어(43) 경계에서는 자동으로
발동하지 않아 원본 의미(마지막 루프 뒤에는 boundary norm이 없고 최종
`output_norm`만 있음)와 일치합니다. **이 패치를 적용 + 재빌드하기 전까지는
`flatten_gguf.py`를 `--accept-boundary-norm-loss` 없이 실행하면 스크립트가
거부합니다.** 급하면 그 플래그로 부정확한 펼침을 만들 수는 있으나, ③ 출력
동일성 검증에서 반드시 실패할 것으로 예상해야 합니다.

## ① Q8_0 변환/다운로드 권고

로컬 BF16(8.34GB)을 양자화하는 대신 HF에 이미 있는 Q8_0을 바로 받는 편이
디스크 I/O와 시간 모두 절약됩니다.

```bash
hf download owao/Nanbeige4.2-3B-GGUF --include "*Q8_0*" \
  --local-dir /Users/angigyeom/Desktop/optillama/models/Nanbeige4.2-3B-GGUF
```

실측 파일 크기 (HF 메타데이터 확인): `nanbeige4.2-3b-Q8_0.gguf` = **4.43 GB**.

펼친판 크기는 폴더(원본) 전체를 그대로 2배 하는 게 아닙니다 — `token_embd`
+ `output`(둘 다 vocab=166144 x n_embd=3072, Q8_0 기준 약 1.08GB)는 루프와
무관하게 1벌만 존재하므로 고정비입니다. 물리 22층(blk.*) 텐서만 Q8_0
기준 약 3.35GB이고 이게 그대로 한 벌 더 늘어납니다. 즉:

```
folded  ≈ 4.43 GB   (blk 3.35GB + embed/output/norm 1.08GB)
flat    ≈ 4.43 + 3.35 = 7.78 GB   (사용자 추정 8.8GB보다 약간 작음)
```

실제 변환 후 `ls -la`로 정확한 값을 다시 확인할 것 (양자화 세부 처리 방식에
따라 ±수백MB 오차 가능).

## ② 펼치기 실행 명령

```bash
# (권장) nanbeige.cpp 패치 + 재빌드 후 — 수학적으로 동일한 펼침
python3 flatten_gguf.py \
  models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0.gguf \
  models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0-flat44.gguf

# (패치 전, 급할 때만) 경계 norm 손실을 인지하고 진행 — ③ 검증 실패 예상
python3 flatten_gguf.py \
  models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0.gguf \
  models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0-flat44.gguf \
  --accept-boundary-norm-loss
```

패치를 적용했다면 재빌드:
```bash
cd nanbeige-llamacpp && cmake --build build-m1max --target llama-cli -j
```

## ③ 두 판의 출력 동일성 검증 명령

1차(필수, 값싼) — 동일 프롬프트 그리디 생성 비교:

```bash
LLAMA_MMAP_NO_PREFETCH=1 build-m1max/bin/llama-cli \
  -m models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0.gguf \
  -p "The story begins" --temp 0 --seed 42 -n 64 -t 8 -fa on -ngl 0 \
  --no-warmup -no-cnv -nr < /dev/null > /tmp/twin_folded.log 2>&1

LLAMA_MMAP_NO_PREFETCH=1 build-m1max/bin/llama-cli \
  -m models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0-flat44.gguf \
  -p "The story begins" --temp 0 --seed 42 -n 64 -t 8 -fa on -ngl 0 \
  --no-warmup -no-cnv -nr < /dev/null > /tmp/twin_flat.log 2>&1

diff <(grep -v '^llama_\|^main:\|load time\|eval time' /tmp/twin_folded.log) \
     <(grep -v '^llama_\|^main:\|load time\|eval time' /tmp/twin_flat.log)
```

`diff`가 비어야 함(로그 타이밍/헤더 줄 제외). 짧은 그리디 출력은 argmax가
안 바뀌는 작은 수치 오차를 가릴 수 있으니, 가능하면 2차로 `llama-perplexity`
로 동일한 짧은 텍스트에 대해 두 파일의 PPL을 비교해 더 촘촘한 수치 일치를
확인할 것 (완전히 동일한 함수라면 PPL도 소수점 단위까지 일치해야 함).

**동일성 검증이 실패하면 그 다음 어떤 측정도 진행하지 말 것** — 다른 함수를
비교하는 것이 되어 버림 (V1 참고).

## ④ 밸러스트 크기 계산

목표: 가용 RAM 창이 `folded 필요치 < 가용 RAM < flat 필요치`가 되도록.

```
folded 필요치 ≈ 4.43GB(가중치) + 여유(컨텍스트/버퍼) ≈ 5.5GB
flat   필요치 ≈ 7.78GB(가중치) + 여유                 ≈ 8.5GB
목표 가용 RAM 창                                        ≈ 6.5~7GB (중간값)
```

측정 직전에 실측:
```bash
FREE_GB=$(python3 -c "
import subprocess
out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
d = {}
for line in out.splitlines():
    if ':' in line:
        k,v = line.split(':'); d[k.strip()] = int(v.strip().rstrip('.') or 0)
page = 16384
free_like = d.get('Pages free',0) + d.get('Pages inactive',0) + d.get('Pages speculative',0)
print(round(free_like*page/1e9, 2))
")
TARGET_AVAIL_GB=6.5
BALLAST_GB=$(python3 -c "print(max(0, $FREE_GB - $TARGET_AVAIL_GB - 1))")  # -1은 OS 안전마진
echo "free-like=${FREE_GB}GB -> ballast=${BALLAST_GB}GB"
```

총 RAM은 32GiB(`sysctl hw.memsize` = 34359738368)이므로 다른 상주
프로세스(다운로드/에이전트 등)가 많으면 `BALLAST_GB`는 그만큼 줄어듦 —
매번 다시 계산할 것, 고정값을 재사용하지 말 것.

밸러스트 실행/해제:
```bash
python3 ballast.py --gb $BALLAST_GB &         # 측정 세트 시작 전
BALLAST_PID=$!
...  # ⑤ 측정
kill -TERM $BALLAST_PID                        # 측정 세트 끝나면 반드시 해제
```

## ⑤ 측정 커맨드 (v5 워치독 패턴 준수)

`e1_battery_v6.sh`와 동일한 안전 규칙을 그대로 따를 것: 동시 실행 llama
프로세스 1개(`ps aux | grep llama-cli`로 사전 확인), `-ngl 0`(Metal 금지),
`llama-bench` 금지, `-t 8`(P코어 수), `-nr`(`--no-repack`),
`LLAMA_MMAP_NO_PREFETCH=1`, `--no-warmup -no-cnv < /dev/null`, 워치독으로
`[ Prompt:` 등장/swap_delta>1500MB/제한시간 초과 시 `kill -9`. 다운로드
병행 금지 — T1(IQ2_XXS) 다운로드가 끝난 뒤에만 시작.

```bash
run() {
  local name=$1 model=$2 maxsec=$3
  sync; sleep 3
  local pi0=$(vm_stat | awk '/Pageins/{gsub(/\./,"");print $2}')
  local sw0=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}')
  local t0=$(date +%s) peak=0 st=TIMEOUT
  local LOG=/tmp/twin_${name}.log
  LLAMA_MMAP_NO_PREFETCH=1 build-m1max/bin/llama-cli -m $model \
    -p "The story begins" -n 32 -t 8 -fa on -ngl 0 \
    --no-warmup -no-cnv -nr < /dev/null > $LOG 2>&1 &
  local PID=$!
  while true; do
    sleep 5
    local rss=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && { st=EXITED; break; }
    [ "$rss" -gt "$peak" ] && peak=$rss
    grep -q '\[ Prompt:' $LOG 2>/dev/null && { st=OK; kill -9 $PID 2>/dev/null; break; }
    local swnow=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%d", $6}')
    [ $((swnow - sw0)) -gt 1500 ] && { st=SWAP_ABORT; kill -9 $PID 2>/dev/null; break; }
    [ $(( $(date +%s) - t0 )) -gt $maxsec ] && { kill -9 $PID 2>/dev/null; break; }
  done
  local perf=$(grep -o '\[ Prompt:.*\]' $LOG | tail -1)
  echo "$name st=$st peakRSS=$((peak/1048576))GB $perf" >> protolab/twin_results.log
}

run folded models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0.gguf       900
run flat   models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q8_0-flat44.gguf 900
```

## ⑥ 예상 결과와 kill-gate

**예상:** folded(≈5.5GB 필요)는 밸러스트로 만든 ≈6.5GB 창 안에 상주 →
정상 속도. flat(≈8.5GB 필요)는 창을 초과 → mmap 폴트로 SSD에서 스트리밍
→ 뚜렷한 t/s 저하와 `ssd_read`/`pageins` 증가.

**kill-gate:**
- ③ 출력 동일성 검증 실패(diff 불일치) → 즉시 중단, 패치부터 재점검. 부정확한 두 함수를 비교하는 벤치마크는 무의미함.
- swap_delta > 1500MB 발생 → 기존 규칙대로 즉시 kill, 밸러스트 과다 설정 의심.
- flat이 folded 대비 유의미하게 느려지지 않음(잡음 범위, 예: ±15% 이내) → "펼친판 스트리밍 절벽" 가설이 이 설정에서는 기각 — 밸러스트 창을 더 좁혀 재시도할지, 아니면 트랙 자체를 접을지 판단 필요. 무리하게 밸러스트를 키워 재시도하지 말고 먼저 결과를 기록.
- peakRSS가 예상 범위(folded ~5-6GB, flat ~8-9GB)에서 크게 벗어나면 밸러스트 계산 재검토.
