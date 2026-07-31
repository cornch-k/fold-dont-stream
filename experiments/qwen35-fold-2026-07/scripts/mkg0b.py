import sys
N_ZERO, out = int(sys.argv[1]), sys.argv[2]
G, N = 7, 48
wl = []
for il in range(N):
    if il >= 45: wl.append(il)
    else:
        g = il * G // 45
        wl.append((g*45 + G - 1)//G)
vals, z = [], 0
for il in range(N):
    folded = (wl[il] != il)
    if folded and il < N_ZERO: g = 0.0; z += 1
    else:                      g = 0.3
    vals.append(g)
open(out,'w').write('\n'.join(f'{v:.4f}' for v in vals) + '\n')
print(f"N<{N_ZERO}: γ=0 층 {z}")
