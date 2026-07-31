#!/usr/bin/env python3
"""Q3_K_S GGUF 에서 MTP 사이드카 추출.
mtp_only 로더(qwen35moe.cpp:45)가 요구하는 것만: token_embd, output_norm, output, blk.48.*
메타데이터는 전부 복사 (block_count 49 / nextn 1 유지 — 로더가 blk.0 부재로 mtp_only 판정)."""
import sys
sys.path.insert(0, '/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/gguf-py')
import gguf

SRC = '/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf'
DST = '/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-mtp-fat.gguf'
KEEP = lambda n: n.startswith('blk.48.') or n in ('token_embd.weight', 'output_norm.weight', 'output.weight')

reader = gguf.GGUFReader(SRC)
arch = None
for f in reader.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE:
        arch = f.contents()
writer = gguf.GGUFWriter(DST, arch)

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

kept = [t for t in reader.tensors if KEEP(t.name)]
total = sum(int(t.n_bytes) for t in kept)
print(f'유지 텐서 {len(kept)}개, {total/1e9:.3f} GB')
for t in kept:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in kept:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.close()
print(f'완료: {DST}')
