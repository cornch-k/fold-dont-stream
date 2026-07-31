import sys
mode, out = sys.argv[1], sys.argv[2]
G, N = 7, 48
wl = []
for il in range(N):
    if il >= 45: wl.append(il)
    else:
        g = il * G // 45
        wl.append((g*45 + G - 1)//G)
vals = []
for il in range(N):
    folded = (wl[il] != il)
    if not folded:            g = 0.3          # 대표/고정층은 그대로
    elif mode == 'half_odd':  g = 0.0 if il % 2 == 1 else 0.3
    elif mode == 'half_front':g = 0.0 if il < 23 else 0.3
    elif mode == 'half_back': g = 0.0 if il >= 23 else 0.3
    elif mode == 'third':     g = 0.0 if il % 3 != 0 else 0.3
    else:                     g = 0.3
    vals.append(g)
open(out,'w').write('\n'.join(f'{v:.4f}' for v in vals) + '\n')
z = sum(1 for v in vals if v == 0.0)
print(f"{mode}: γ=0 층 {z}/48")
