#!/usr/bin/env python3
"""
flatten_gguf.py - physically unroll a looped-transformer GGUF into a flat,
non-looped GGUF of the same total logical depth ("flattened twin").

Target model: Nanbeige4.2-3B (22 physical blk.* layers, {arch}.num_loops=2 ->
44 logical layers). This script duplicates blk.0..blk.{n_phys-1} into
blk.{n_phys}..blk.{2*n_phys-1} (and further loop slots if num_loops > 2),
rewrites block_count to n_phys*num_loops and num_loops to 1, and copies every
other KV/tensor untouched.

IMPORTANT - read before using (see V1 finding in flattened-twin-protocol.md):

llama_model_nanbeige::graph() (nanbeige-llamacpp/src/models/nanbeige.cpp)
inserts one extra RMSNorm, reusing the *output_norm* weights, at every loop
boundary:

    if (n_loops > 1 &&
        ((il + 1) % n_phys) == 0 &&
        (il + 1) < n_layer &&
        !nb.skip_loop_final_norm) {
        cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, il);
        ...
    }

That insertion is gated on `n_loops > 1`. Setting num_loops=1 in the
flattened file's metadata (what this script does, per spec) makes llama.cpp
skip this norm *unconditionally* - it does not matter that block_count grew
to 44. If the source file has skip_loop_final_norm=false (true for the
current Nanbeige4.2-3B-GGUF release, confirmed by reading its own KV), the
folded model actually executes that extra norm once (between physical layer
21 and physical layer 22), and a naive metadata-only flatten omits it. The
two "twins" would then NOT be computing the same function.

This script will refuse to produce such a file unless you pass
--accept-boundary-norm-loss, acknowledging the discrepancy. The correct fix
is a small nanbeige.cpp patch that reinserts the norm at the physical-layer
boundary even when num_loops==1, e.g. keyed off the
"{arch}.flatten.source_block_count" KV this script writes into the output
file for exactly that purpose. See flattened-twin-protocol.md for the
patch sketch. That patch/rebuild is NOT applied by this script - it only
edits the GGUF.

Usage:
    python3 flatten_gguf.py INPUT.gguf OUTPUT.gguf [--force]
                            [--accept-boundary-norm-loss] [--verbose]

Only metadata + a tensor-info/copy plan are built here; nothing is read
"large" until write_tensor_data() actually streams bytes through, which only
happens when this script is *run* (not merely imported/compiled).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Prefer the gguf-py bundled in this repo's local llama.cpp fork; fall back
# to whatever "gguf" is importable (e.g. pip-installed) otherwise.
_LOCAL_GGUF = Path(__file__).resolve().parent / "nanbeige-llamacpp" / "gguf-py"
if _LOCAL_GGUF.exists() and "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(_LOCAL_GGUF))

import gguf  # noqa: E402

logger = logging.getLogger("flatten-gguf")


def get_field_value(reader: "gguf.GGUFReader", key: str, default=None):
    field = reader.get_field(key)
    if field is None:
        return default
    return field.contents()


def per_layer_suffix_counts(reader: "gguf.GGUFReader", n_phys: int) -> dict[str, int]:
    """For every 'blk.<i>.<suffix>' tensor with i in [0, n_phys), count how
    many of the n_phys physical indices actually carry that suffix.

    Tensors created with TENSOR_DUPLICATED for i != 0 (e.g. rope_freqs, see
    nanbeige.cpp create_arch_tensors()) only exist at i==0 in the file; a
    suffix with count == 1 (and only at i==0) is such a "loop-shared" tensor
    and must NOT be duplicated into the new loop slots - the loader expects
    it to be absent for every physical index other than 0, exactly like in
    the source file.
    """
    counts: dict[str, int] = {}
    for t in reader.tensors:
        if not t.name.startswith("blk."):
            continue
        rest = t.name[len("blk."):]
        idx_str, sep, suffix = rest.partition(".")
        if not sep or not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if idx >= n_phys:
            continue  # already a higher-loop slot (e.g. re-flattening a flat file)
        counts[suffix] = counts.get(suffix, 0) + 1
    return counts


def build_duplication_plan(reader: "gguf.GGUFReader", n_phys: int, n_loops: int,
                            always_present: set[str]) -> list[tuple[str, "gguf.ReaderTensor"]]:
    plan: list[tuple[str, "gguf.ReaderTensor"]] = []
    for tensor in reader.tensors:
        if not tensor.name.startswith("blk."):
            continue
        rest = tensor.name[len("blk."):]
        idx_str, sep, suffix = rest.partition(".")
        if not sep or not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if idx >= n_phys or suffix not in always_present:
            continue
        for k in range(1, n_loops):
            new_name = f"blk.{idx + k * n_phys}.{suffix}"
            plan.append((new_name, tensor))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physically unroll a looped-transformer GGUF (Nanbeige-style "
                    "{arch}.num_loops) into a flat GGUF of the same logical depth.")
    parser.add_argument("input", type=Path, help="source GGUF (folded, num_loops > 1)")
    parser.add_argument("output", type=Path, help="destination GGUF (flat, num_loops = 1)")
    parser.add_argument("--force", action="store_true",
                         help="overwrite output if it already exists")
    parser.add_argument("--accept-boundary-norm-loss", action="store_true",
                         help="proceed even though skip_loop_final_norm=false in the "
                              "source file, i.e. the flattened output will NOT be "
                              "mathematically identical to the folded model unless "
                              "nanbeige.cpp is patched to reinsert the loop-boundary "
                              "norm(s) when loading it (see flattened-twin-protocol.md)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(levelname)s: %(message)s")

    if not args.input.exists():
        logger.error(f"input file not found: {args.input}")
        sys.exit(1)

    if args.output.exists() and not args.force:
        logger.error(f"{args.output} already exists, pass --force to overwrite")
        sys.exit(1)

    logger.info(f"reading {args.input}")
    reader = gguf.GGUFReader(args.input, "r")

    arch = get_field_value(reader, gguf.Keys.General.ARCHITECTURE)
    if arch != "nanbeige":
        logger.warning(f"general.architecture = {arch!r}, expected 'nanbeige' - "
                        f"proceeding, but the {{arch}}.num_loops / "
                        f"skip_loop_final_norm handling below is nanbeige-specific "
                        f"and may not apply to this architecture")

    n_phys = get_field_value(reader, gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch))
    n_loops = get_field_value(reader, gguf.Keys.LLM.NUM_LOOPS.format(arch=arch), 1) or 1
    skip_loop_final_norm = bool(get_field_value(
        reader, gguf.Keys.LLM.SKIP_LOOP_FINAL_NORM.format(arch=arch), False))

    if n_phys is None:
        logger.error("could not read {arch}.block_count from source file")
        sys.exit(1)

    logger.info(f"arch={arch} block_count(n_phys)={n_phys} num_loops={n_loops} "
                f"skip_loop_final_norm={skip_loop_final_norm}")

    if n_loops <= 1:
        logger.error("source num_loops <= 1: file is already flat, nothing to do")
        sys.exit(1)

    if not skip_loop_final_norm and not args.accept_boundary_norm_loss:
        logger.error(
            "skip_loop_final_norm=false in the source file: the folded model "
            "inserts an extra RMSNorm (output_norm weights) at every loop "
            "boundary (see llama_model_nanbeige::graph() in "
            "nanbeige-llamacpp/src/models/nanbeige.cpp - the "
            "'n_loops > 1 && ... && !skip_loop_final_norm' branch). Writing "
            "num_loops=1 into the flattened file makes llama.cpp skip that "
            "norm unconditionally, since the check is `n_loops > 1`, not a "
            "check on block_count. This flatten would therefore NOT be "
            "mathematically identical to the folded model as-is - a small "
            "nanbeige.cpp patch is needed to reinsert the boundary norm(s) at "
            "the physical-layer boundary even when num_loops==1 (see "
            "flattened-twin-protocol.md). Re-run with "
            "--accept-boundary-norm-loss to write the (inexact) flattened "
            "file anyway.")
        sys.exit(1)

    new_n_layer = n_phys * n_loops

    suffix_counts = per_layer_suffix_counts(reader, n_phys)
    always_present = {s for s, c in suffix_counts.items() if c == n_phys}
    loop_shared = {s for s, c in suffix_counts.items() if c != n_phys}
    if loop_shared:
        logger.info("not duplicating loop-shared per-layer tensors (present at only "
                    f"some physical indices, e.g. rope_freqs at i=0 only): "
                    f"{sorted(loop_shared)}")

    dup_plan = build_duplication_plan(reader, n_phys, n_loops, always_present)

    logger.info(f"writing {args.output} "
                f"(block_count {n_phys} -> {new_n_layer}, num_loops {n_loops} -> 1, "
                f"{len(dup_plan)} tensors duplicated)")

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    block_count_key = gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch)
    num_loops_key = gguf.Keys.LLM.NUM_LOOPS.format(arch=arch)

    for field in reader.fields.values():
        # general.architecture is written by the GGUFWriter constructor already;
        # GGUF.* are virtual fields synthesized by the reader, not real KVs.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()

        if field.name == block_count_key:
            value = new_n_layer
        elif field.name == num_loops_key:
            value = 1

        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    # Provenance KVs: harmless to loaders that don't know them, and exactly
    # what a future nanbeige.cpp patch would need to reinsert the dropped
    # boundary norm(s) at the right physical-layer stride.
    writer.add_key_value(f"{arch}.flatten.source_block_count", int(n_phys), gguf.GGUFValueType.UINT32)
    writer.add_key_value(f"{arch}.flatten.source_num_loops", int(n_loops), gguf.GGUFValueType.UINT32)
    writer.add_key_value(f"{arch}.flatten.boundary_norm_dropped", not skip_loop_final_norm, gguf.GGUFValueType.BOOL)

    tensor_plan: list[tuple[str, "gguf.ReaderTensor"]] = [(t.name, t) for t in reader.tensors]
    tensor_plan.extend(dup_plan)

    total_bytes = sum(t.n_bytes for _, t in tensor_plan)
    logger.info(f"{len(tensor_plan)} tensors total "
                f"({len(dup_plan)} new), {total_bytes / 1e9:.2f} GB to write")

    for name, tensor in tensor_plan:
        writer.add_tensor_info(name, tensor.data.shape, tensor.data.dtype,
                                tensor.data.nbytes, tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for name, tensor in tensor_plan:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

    writer.close()
    logger.info("done")


if __name__ == "__main__":
    main()
