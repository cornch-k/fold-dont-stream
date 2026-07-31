#!/usr/bin/env python3
"""EXP-56 생성형 MMLU 프로브: 우도 고정(33)이 생성 모드에서도 유지되는가.
서버(A 구성)에 100문항을 '문자만 답하라'로 묻고 채점."""
import json, urllib.request, re, sys, time

rows = [r["row"] for r in json.load(open("exp/mmlu100.json"))["rows"]]
URL = "http://127.0.0.1:8087/completion"
LET = "ABCD"
ok = 0; n = 0; fails = 0
t0 = time.time()
for i, r in enumerate(rows):
    ch = r["choices"]
    prompt = (f"Question: {r['question']}\n"
              + "\n".join(f"{LET[j]}) {c}" for j, c in enumerate(ch))
              + "\nAnswer with only the letter (A, B, C, or D).\nAnswer:")
    body = json.dumps({
        "prompt": prompt, "n_predict": 4, "temperature": 0,
        "logit_bias": [[248068, -20]],  # <think> 차단
    }).encode()
    try:
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        text = resp.get("content", "")
    except Exception as e:
        fails += 1
        if fails > 10: sys.exit(f"서버 실패 과다: {e}")
        continue
    m = re.search(r"[ABCD]", text.upper())
    n += 1
    if m and LET.index(m.group(0)) == r["answer"]: ok += 1
    if (i+1) % 20 == 0:
        print(f"{i+1}문항: 정답률 {ok/max(n,1)*100:.1f}%  ({time.time()-t0:.0f}s)")
print(f"최종: {ok}/{n} = {ok/max(n,1)*100:.1f}%   (우도 채점 33.2, 우연 25)")
