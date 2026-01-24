"""
Small collection of PyTree utilities. 
- They assume *all leaves are JAX arrays* (jnp.ndarray).
- They treat the *leading axis* as a batch / time dimension.
- They are designed to work cleanly under `jax.jit`, `jax.vmap`, and `jax.lax.scan` / `while_loop`.

Typical use cases:
- Preallocating a “batched” structure (e.g. history of kernel params or MCMC stats) given a single example instance.
- Writing a single unbatched instance into a batched tree at a specific index.
- Truncating a batched tree to the first `b` entries (dynamic slice).
- Reducing a batched tree across the leading axis (e.g. average across chains).

All functions operate on arbitrary PyTrees: dataclasses decorated with `flax.struct.dataclass`,
tuples, dicts, lists, etc., as long as the leaves are arrays with compatible shapes.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

PyTree = Any


def tree_make_batched(example: PyTree, B: int) -> PyTree:
    """
    Allocate a *batched* PyTree of zeros from an unbatched example.

    This is the generic replacement for `make_dataclass_batched`. Given a
    representative instance (e.g. a single kernel params struct or stats
    summary), it constructs a new PyTree of the same structure where each
    leaf is a zero array with an added leading batch dimension of size `B`.

    Concretely, for each leaf `leaf` (converted to `jnp.asarray(leaf)`):

        leaf.shape      -> (d1, d2, ...)
        new_leaf.shape  -> (B, d1, d2, ...)

    Parameters
    ----------
    example:
        Any PyTree whose leaves are arrays or array-like. This is used purely
        to infer the tree structure, shapes, and dtypes.
    B:
        Integer batch size to allocate. May be a Python int or a JAX scalar.

    Returns
    -------
    batched_example:
        A new PyTree with the same structure as `example`, where every leaf
        is a zero array with an extra leading dimension of size `B`.

    Notes
    -----
    - This is JIT-safe: it only uses `jnp.zeros` and `tree_map`.
    - It does *not* copy values from `example`; it only uses its shapes/dtypes.
    - Typical use: preallocate histories of length `max_steps` or `n_time`.
    """
    def promote(leaf):
        arr = jnp.asarray(leaf)
        return jnp.zeros((B,) + arr.shape, dtype=arr.dtype)

    return jax.tree_util.tree_map(promote, example)


def tree_update_at_index(
    batched_tree: PyTree,
    single_tree: PyTree,
    index: int,
) -> PyTree:
    """
    Insert/update a single unbatched PyTree into a batched PyTree at a given index.

    This is the generic replacement for `update_dataclass_at_index` /
    `insert_dataclass_instance_at_index`. It assumes `batched_tree` has a
    leading batch axis and that `single_tree` has the same structure without
    that batch axis.

    For each pair of corresponding leaves `(batched_leaf, single_leaf)`:

        batched_leaf.shape == (B, ...)  # B >= index+1
        single_leaf.shape  == (...)

    we perform:

        batched_leaf_new = batched_leaf.at[index].set(single_leaf)

    Parameters
    ----------
    batched_tree:
        PyTree whose leaves are arrays with shape `(B, ...)`. This is the
        “history” or batched storage.
    single_tree:
        PyTree of the same structure as `batched_tree`, but with leaves that
        match the *non-batch* shape: `single_leaf.shape == batched_leaf.shape[1:]`.
    index:
        Integer index into the leading dimension at which `single_tree` should
        be written.

    Returns
    -------
    updated_batched_tree:
        A new PyTree with the same structure as `batched_tree`, where each leaf
        has been updated at position `index`.

    Notes
    -----
    - This is JIT- and `while_loop`-friendly: it uses `.at[index].set(...)` inside a `tree_map`.
    - There are no explicit checks on shapes; mismatches will fail at runtime.
    """
    def upd(batched_leaf, single_leaf):
        return batched_leaf.at[index].set(single_leaf)

    return jax.tree_util.tree_map(upd, batched_tree, single_tree)


def tree_truncate_leading(batched_tree: PyTree, b: int) -> PyTree:
    """
    Truncate a batched PyTree along the leading axis to length `b`.

    This is the generic replacement for `truncate_dataclass`. It keeps only
    the first `b` entries along axis 0 for each leaf, using `lax.dynamic_slice`
    to stay JIT-compatible with dynamic `b`.

    For each leaf `leaf`:

        leaf.shape      == (T, d1, d2, ...)
        truncated.shape == (b, d1, d2, ...)

    Parameters
    ----------
    batched_tree:
        PyTree whose leaves are arrays with at least one dimension. The first
        dimension is interpreted as batch/time.
    b:
        Number of leading entries to keep. May be a Python int or a JAX scalar.
        Must satisfy `0 <= b <= T` where `T` is the original size.

    Returns
    -------
    truncated_tree:
        A new PyTree with the same structure as `batched_tree`, where each leaf
        has been truncated to the first `b` entries along the leading axis.

    Notes
    -----
    - This is useful when you allocate histories to a maximum length
    (`max_steps`) but stop early based on a convergence or temperature
    condition and want to slice them down to the actual number of steps.
    """
    def trunc(leaf):
        arr = jnp.asarray(leaf)
        shape = arr.shape
        if arr.ndim == 0:
            # Scalar leaf: nothing to truncate; return as-is.
            return arr
        slice_sizes = (b,) + shape[1:]
        return lax.dynamic_slice(arr, (0,) * arr.ndim, slice_sizes)

    return jax.tree_util.tree_map(trunc, batched_tree)


def tree_mean_across_batch(batched_tree: PyTree) -> PyTree:
    """
    Compute the mean across the leading (batch) axis for every leaf in a PyTree.

    This is the generic replacement for `mean_across_batch` on dataclasses.
    It assumes each leaf has a leading batch dimension and reduces that
    dimension with `jnp.mean(axis=0)`.

    For each leaf `leaf`:

        leaf.shape      == (B, d1, d2, ...)
        mean_leaf.shape == (d1, d2, ...)

    Parameters
    ----------
    batched_tree:
        PyTree whose leaves are arrays with shape `(B, ...)`. This might be,
        for example, a collection of per-chain MCMC summaries or per-particle
        statistics.

    Returns
    -------
    mean_tree:
        PyTree of the same structure, where each leaf is the mean across
        the leading axis of the corresponding leaf in `batched_tree`.

    Notes
    -----
    - Common usage:
        - Turn per-chain MCMC summaries (shape `(n_chains, ...)`) into a
          single average summary.
        - Aggregate stats from a batched run (e.g. in `run_mcmc_batch`)
          into a single representative summary.
    """
    def mean_leaf(leaf):
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            # Scalar leaf: nothing to average; return as-is.
            return arr
        return jnp.mean(arr, axis=0)

    return jax.tree_util.tree_map(mean_leaf, batched_tree)