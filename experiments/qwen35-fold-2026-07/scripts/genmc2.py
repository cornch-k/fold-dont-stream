#!/usr/bin/env python3
"""EXP-66 유효한 생성형 MMLU: Qwen3.5 는 thinking 기본 모델이므로
채팅 템플릿 + <think> 허용 + 충분한 출력 길이로 재야 한다.
이전 EXP-56 은 <think> 를 차단해서 무효였다(27%).
공식 MMLU-Redux 94.0(미접힘) / 35B 93.3 과 같은 계열의 측정."""
import json, urllib.request, re, sys, time, argparse

AP = argparse.ArgumentParser()
AP.add_argument("--n", type=int, default=100)
AP.add_argument("--max-tokens", type=int, default=2048)
AP.add_argument("--port", type=int, default=8087)
AP.add_argument("--no-think", action="store_true")
A = AP.parse_args()

rows = [r["row"] for r in json.load(open("exp/mmlu100.json"))["rows"]][:A.n]
URL = f"http://127.0.0.1:{A.port}/v1/chat/completions"
LET = "ABCD"
ok = n = fails = 0
tok_used = []
t0 = time.time()
for i, r in enumerate(rows):
    ch = r["choices"]
    q = (f"{r['question']}\n\n"
         + "\n".join(f"{LET[j]}) {c}" for j, c in enumerate(ch))
         + "\n\nReason step by step, then put your final answer letter within \\boxed{}.")
    body = {"messages": [{"role": "user", "content": q}],
            "max_tokens": A.max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    if A.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        text = resp["choices"][0]["message"]["content"] or ""
        tok_used.append(resp.get("usage", {}).get("completion_tokens", 0))
    except Exception as e:
        fails += 1
        if fails > 8: sys.exit(f"서버 실패 과다: {e}")
        continue
    # \boxed{X} 우선, 없으면 마지막에 등장한 단독 letter
    m = re.findall(r"boxed\{\s*\(?([ABCD])", text)
    if not m:
        m = re.findall(r"(?:answer|Answer)\D{0,12}([ABCD])\b", text)
    if not m:
        m = re.findall(r"\b([ABCD])\b", text[-200:])
    n += 1
    if m and LET.index(m[-1]) == r["answer"]: ok += 1
    if (i + 1) % 10 == 0:
        print(f"{i+1}문항: {ok/max(n,1)*100:.1f}%  평균출력 {sum(tok_used)/max(len(tok_used),1):.0f}토큰  ({time.time()-t0:.0f}s)", flush=True)
print(f"최종: {ok}/{n} = {ok/max(n,1)*100:.1f}%   (우도채점 33.4, thinking차단 27.0, 우연 25)")
