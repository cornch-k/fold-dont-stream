#!/usr/bin/env python3
"""EXP-66b 예산 강제(budget forcing) 생성형 MMLU.

EXP-66 실패 원인: 접힌 모델이 추론을 종료하지 못해 토큰 상한(2048)까지 생각만 하고
최종 답을 내지 않았다 → 추출 실패로 10%(우연 25% 미달). 무효 측정이었다.

해결: 2단계로 답을 강제한다.
  1단계  추론을 THINK 토큰까지만 허용
  2단계  1단계 출력 뒤에 "</think>\n\n정답:" 을 붙여 답 토큰만 생성
llama.cpp /completion 을 쓰고 채팅 템플릿은 직접 구성한다(서버 템플릿 의존 제거).
"""
import json, urllib.request, re, sys, time, argparse

AP = argparse.ArgumentParser()
AP.add_argument("--n", type=int, default=60)
AP.add_argument("--think", type=int, default=600, help="추론 토큰 예산")
AP.add_argument("--port", type=int, default=8087)
AP.add_argument("--no-think", action="store_true", help="추론 없이 즉답(대조군)")
A = AP.parse_args()

rows = [r["row"] for r in json.load(open("exp/mmlu100.json"))["rows"]][:A.n]
URL = f"http://127.0.0.1:{A.port}/completion"
LET = "ABCD"


def post(prompt, n_predict, stop=None):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "cache_prompt": True}
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r.get("content", ""), r.get("tokens_predicted", 0)


ok = n = fails = 0
tot_tok = 0
t0 = time.time()
for i, r in enumerate(rows):
    ch = r["choices"]
    q = (f"{r['question']}\n"
         + "\n".join(f"{LET[j]}) {c}" for j, c in enumerate(ch))
         + "\nAnswer with the single correct letter.")
    # Qwen3.5 채팅 템플릿을 직접 구성
    head = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    try:
        if A.no_think:
            body = head + "<think>\n\n</think>\n\nAnswer: \\boxed{"
            out, t = post(body, 6, stop=["}", "\n"])
            tot_tok += t
        else:
            think, t1 = post(head + "<think>\n", A.think, stop=["</think>"])
            # 2단계: 추론을 강제 종료시키고 답만 받는다
            forced = head + "<think>\n" + think + "\n</think>\n\nAnswer: \\boxed{"
            out, t2 = post(forced, 6, stop=["}", "\n"])
            tot_tok += t1 + t2
    except Exception as e:
        fails += 1
        if fails > 8:
            sys.exit(f"서버 실패 과다: {e}")
        continue
    m = re.findall(r"[ABCD]", out.upper())
    n += 1
    if m and LET.index(m[0]) == r["answer"]:
        ok += 1
    if (i + 1) % 10 == 0:
        print(f"{i+1}문항: {ok/max(n,1)*100:.1f}%  누적토큰 {tot_tok}  ({time.time()-t0:.0f}s)", flush=True)

mode = "추론없음(대조)" if A.no_think else f"추론예산 {A.think}"
print(f"RESULT66 {mode}: {ok}/{n} = {ok/max(n,1)*100:.1f}%   (우도채점 33.4, 우연 25)")
