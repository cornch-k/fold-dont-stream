#!/usr/bin/env python3
"""EXP-43b 핫 전문가 수술: v2a 전체 복사 + 18개 접힌 층에 핫 텐서 4종 주입."""
import sys
sys.path.insert(0, '/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/gguf-py')
import gguf, numpy as np

SRC = '/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf'
DST = '/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hotF.gguf'

hot = {}
for line in open('/Users/angigyeom/Desktop/DEV/fun/accelerator/exp/hotF_ids.txt'):
    p = line.split()
    hot[int(p[0])] = np.array(sorted(int(x) for x in p[1:]))
print(f"핫 층 {len(hot)}개, 층당 {len(next(iter(hot.values())))}개")

reader = gguf.GGUFReader(SRC)
arch = None
for f in reader.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE: arch = f.contents()
writer = gguf.GGUFWriter(DST, arch)
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'): continue
    vt = field.types[0]
    st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), vt, st and st or None, sub_type=st) if False else writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

src = {t.name: t for t in reader.tensors}
new = []  # (name, data, raw_dtype)
for il, ids in sorted(hot.items()):
    for role, newname in (("gate","ffn_gate_hexps"), ("up","ffn_up_hexps"), ("down","ffn_down_hexps")):
        t = src[f"blk.{il}.ffn_{role}_exps.weight"]
        sl = np.ascontiguousarray(t.data[ids])
        new.append((f"blk.{il}.{newname}.weight", sl, t.tensor_type))
    g = src[f"blk.{il}.ffn_gate_inp.weight"]
    sl = np.ascontiguousarray(g.data[ids])
    new.append((f"blk.{il}.ffn_hot_inp.weight", sl, g.tensor_type))

for t in reader.tensors:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
for name, d, tt in new:
    writer.add_tensor_info(name, d.shape, d.dtype, d.nbytes, tt)

writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
done = 0
for t in reader.tensors:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess); done += 1
    if done % 200 == 0: print(f"  복사 {done}/{len(reader.tensors)}")
for name, d, tt in new:
    writer.write_tensor_data(d, tensor_endianess=reader.endianess)
writer.close()
print(f"완료: {DST}  (+{len(new)}텐서, +{sum(d.nbytes for _,d,_ in new)/1e9:.3f} GB)")
