#!/usr/bin/env python3
"""
ballast.py - allocate and pin N GiB of RAM so that available memory can be
dialed to sit between two model sizes (e.g. below a "flattened twin" GGUF's
footprint, above the folded original's) for a streaming-vs-resident
comparison.

Usage:
    python3 ballast.py --gb 40                 # allocate, mlock, hold until SIGTERM/SIGINT
    python3 ballast.py --gb 1 --test            # self-test: alloc, mlock, verify, release, exit

If mlock() fails (RLIMIT_MEMLOCK too low, or RLIMIT_MEMLOCK not exposed at
all - normal on macOS), falls back to a periodic touch loop that keeps
rewriting one byte per page so the pages stay resident. This is a
best-effort substitute, NOT a real memory pin: the OS can still reclaim
pages between touches under heavy memory pressure.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import signal
import sys
import time

PAGE_SIZE = mmap.PAGESIZE


def _get_libc() -> ctypes.CDLL:
    name = ctypes.util.find_library("c")
    if name is None:
        name = "libc.dylib" if sys.platform == "darwin" else "libc.so.6"
    try:
        return ctypes.CDLL(name, use_errno=True)
    except OSError:
        # CDLL(None) loads symbols already linked into the running process
        # (libSystem/libc on macOS, glibc on Linux) - a portable last resort.
        return ctypes.CDLL(None, use_errno=True)


def touch(buf: mmap.mmap, size: int) -> None:
    """Write one byte per page so every page is physically committed
    (anonymous mmap pages are lazily backed and read as zero until written)."""
    for offset in range(0, size, PAGE_SIZE):
        buf[offset] = 1


def try_raise_memlock_rlimit(size: int) -> str:
    try:
        import resource
    except ImportError:
        return "resource module unavailable"

    if not hasattr(resource, "RLIMIT_MEMLOCK"):
        # Normal on macOS: RLIMIT_MEMLOCK is not exposed by the resource
        # module at all. mlock() there is governed by other kernel limits
        # instead, so just try mlock() directly and see what happens.
        return "RLIMIT_MEMLOCK not defined on this platform (expected on macOS)"

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if hard == resource.RLIM_INFINITY or hard >= size:
            new_soft = size
        else:
            new_soft = hard  # can't exceed hard limit without extra privilege
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
        return f"RLIMIT_MEMLOCK: soft {soft} -> {new_soft} (hard={hard})"
    except (ValueError, OSError) as e:
        return f"could not raise RLIMIT_MEMLOCK: {e}"


def try_mlock(libc: ctypes.CDLL, buf: mmap.mmap, size: int) -> tuple[bool, str]:
    addr = ctypes.addressof((ctypes.c_char * size).from_buffer(buf))
    rc = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    if rc == 0:
        return True, "mlock() succeeded"
    err = ctypes.get_errno()
    return False, f"mlock() failed: {os.strerror(err)} (errno {err})"


def try_munlock(libc: ctypes.CDLL, buf: mmap.mmap, size: int) -> None:
    addr = ctypes.addressof((ctypes.c_char * size).from_buffer(buf))
    libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))


def periodic_touch_fallback(buf: mmap.mmap, size: int, interval: float, stop_flag: list) -> None:
    print(f"[ballast] fallback: re-touching {size / (1024**3):.2f} GiB every "
          f"{interval:.0f}s to discourage eviction (best-effort, not a real pin)")
    while not stop_flag[0]:
        touch(buf, size)
        waited = 0.0
        while waited < interval and not stop_flag[0]:
            time.sleep(min(1.0, interval - waited))
            waited += 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allocate and pin N GiB of RAM as memory-pressure ballast.")
    parser.add_argument("--gb", type=float, required=True,
                         help="GiB (2**30 bytes) to allocate")
    parser.add_argument("--test", action="store_true",
                         help="self-test: allocate, mlock, verify, release, exit immediately "
                              "(does not wait for a signal)")
    parser.add_argument("--touch-interval", type=float, default=30.0,
                         help="seconds between re-touch passes in the no-mlock fallback "
                              "(default: 30)")
    parser.add_argument("--no-mlock", action="store_true",
                         help="skip mlock() entirely and force the touch fallback path "
                              "(useful to exercise/test the fallback)")
    args = parser.parse_args()

    size = int(args.gb * (1024 ** 3))
    size -= size % PAGE_SIZE  # round down to a whole number of pages
    if size <= 0:
        print("nothing to allocate (--gb too small)", file=sys.stderr)
        sys.exit(1)

    print(f"[ballast] allocating {size / (1024**3):.3f} GiB "
          f"({size} bytes, page size {PAGE_SIZE})")
    buf = mmap.mmap(-1, size)

    t0 = time.time()
    touch(buf, size)
    print(f"[ballast] touched all pages in {time.time() - t0:.1f}s (now resident)")

    locked = False
    reason = "skipped (--no-mlock)"
    if not args.no_mlock:
        print(f"[ballast] {try_raise_memlock_rlimit(size)}")
        libc = _get_libc()
        locked, reason = try_mlock(libc, buf, size)
    print(f"[ballast] mlock: {'OK' if locked else 'NOT LOCKED'} ({reason})")
    if not args.no_mlock and not locked:
        print("[ballast] falling back to periodic-touch residency strategy")

    stop_flag = [False]

    def cleanup(signum=None, frame=None):
        stop_flag[0] = True
        if locked:
            try:
                try_munlock(_get_libc(), buf, size)
            except Exception as e:
                print(f"[ballast] munlock failed (ignored): {e}", file=sys.stderr)
        buf.close()
        print("[ballast] released, exiting")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    if args.test:
        print("[ballast] --test mode: releasing immediately")
        cleanup()
        return

    if locked:
        print("[ballast] holding ballast, waiting for SIGTERM/SIGINT ...")
    while True:
        if locked:
            signal.pause()
        else:
            periodic_touch_fallback(buf, size, args.touch_interval, stop_flag)
        if stop_flag[0]:
            break


if __name__ == "__main__":
    main()
