"""Chunk planning helpers for overlap-aware constrained diffusion."""

import math
from itertools import product

import numpy as np


def calculate_overlap_pixels(scale, overlap_factor=2.0):
    """Return overlap halo size in pixels for a target scale."""
    if float(scale) <= 0:
        raise ValueError("scale must be > 0.")
    if float(overlap_factor) <= 0:
        raise ValueError("overlap_factor must be > 0.")
    return int(math.ceil(float(overlap_factor) * float(scale)))


def _prime_factors(n):
    factors = []
    x = int(n)
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def _split_counts_for_n_chunk(shape, n_chunk):
    counts = [1] * len(shape)
    for f in sorted(_prime_factors(int(n_chunk)), reverse=True):
        idx = int(np.argmax([shape[i] / float(counts[i]) for i in range(len(shape))]))
        counts[idx] *= int(f)
    return tuple(counts)


def _build_chunks(data_shape, split_counts, overlap_pixels):
    bins_per_dim = []
    shape = tuple(int(d) for d in data_shape)

    for dim, count in zip(shape, split_counts):
        if int(count) > int(dim):
            raise ValueError(f"Requested split count {count} is larger than dim size {dim}.")
        idx_splits = np.array_split(np.arange(dim), int(count))
        bins = [(int(seg[0]), int(seg[-1]) + 1) for seg in idx_splits if len(seg) > 0]
        bins_per_dim.append(bins)

    chunks = []
    for bins in product(*bins_per_dim):
        output_starts = tuple(b[0] for b in bins)
        output_ends = tuple(b[1] for b in bins)
        chunk_starts = tuple(max(0, s - overlap_pixels) for s in output_starts)
        chunk_ends = tuple(min(dim, e + overlap_pixels) for e, dim in zip(output_ends, shape))
        valid_starts = tuple(s - cs for s, cs in zip(output_starts, chunk_starts))
        valid_ends = tuple(vs + (oe - os) for vs, oe, os in zip(valid_starts, output_ends, output_starts))
        chunks.append(
            {
                "chunk_slices": tuple(slice(cs, ce) for cs, ce in zip(chunk_starts, chunk_ends)),
                "valid_slices": tuple(slice(vs, ve) for vs, ve in zip(valid_starts, valid_ends)),
                "output_slices": tuple(slice(os, oe) for os, oe in zip(output_starts, output_ends)),
            }
        )
    return chunks


def _estimate_chunk_bytes(shape, split_counts, overlap_pixels, itemsize, memory_factor):
    est_elements = 1
    for dim, n_split in zip(shape, split_counts):
        block = int(math.ceil(float(dim) / float(n_split)))
        halo_block = min(int(dim), int(block + 2 * overlap_pixels))
        est_elements *= max(1, halo_block)
    return int(est_elements * int(itemsize) * float(memory_factor))


def plan_vram_chunks(
    data_shape,
    max_scale,
    vram_limit_gb,
    overlap_factor=2.0,
    dtype=np.float32,
    memory_factor=4.0,
    max_chunks=4096,
):
    """
    Plan overlap-aware chunks that fit within a VRAM budget.

    Returns a dict with a chunk list and planning metadata.
    """
    shape = tuple(int(d) for d in data_shape)
    if len(shape) == 0 or any(d <= 0 for d in shape):
        raise ValueError("data_shape must contain positive dimensions.")
    if float(vram_limit_gb) <= 0:
        raise ValueError("vram_limit_gb must be > 0.")
    if float(memory_factor) <= 0:
        raise ValueError("memory_factor must be > 0.")

    itemsize = np.dtype(dtype).itemsize
    budget_bytes = int(float(vram_limit_gb) * (1024 ** 3))
    overlap_pixels = calculate_overlap_pixels(max_scale, overlap_factor=overlap_factor)

    n_chunk = 1
    split_counts = _split_counts_for_n_chunk(shape, n_chunk)
    est_chunk_bytes = _estimate_chunk_bytes(shape, split_counts, overlap_pixels, itemsize, memory_factor)

    while est_chunk_bytes > budget_bytes and n_chunk < int(max_chunks):
        n_chunk += 1
        split_counts = _split_counts_for_n_chunk(shape, n_chunk)
        est_chunk_bytes = _estimate_chunk_bytes(shape, split_counts, overlap_pixels, itemsize, memory_factor)

    chunks = _build_chunks(shape, split_counts, overlap_pixels)
    return {
        "n_chunk": int(len(chunks)),
        "split_counts": tuple(int(v) for v in split_counts),
        "overlap_pixels": int(overlap_pixels),
        "estimated_chunk_bytes": int(est_chunk_bytes),
        "budget_bytes": int(budget_bytes),
        "chunks": chunks,
    }
