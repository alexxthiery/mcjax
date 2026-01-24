"""
Utilities for working with batched `flax.struct.dataclass`.

This module provides functions to:
- Create a batched version of a dataclass, allocating a leading batch dimension.
- Insert a single (unbatched) dataclass instance into a batched version at a specified index.
- Truncate the batch dimension of a dataclass to a given size.

Assumptions:
- The input dataclasses must be decorated with `@flax.struct.dataclass`.
- All fields must be `jax.numpy.ndarray` (`jnp.ndarray`) types.
- Field shapes must support stacking via `jnp.zeros`, slicing, and indexed updates.
"""
from flax import struct
import jax.numpy as jnp
from jax import lax
from dataclasses import fields 

    
def make_dataclass_batched(dataclass_instance: struct.dataclass, B: int) -> struct.dataclass:
    """
    JAX-compatible creation of a batched struct.dataclass.
    Promotes scalar fields to shape (B,) arrays. Works under `jax.jit`.
    """
    def promote(field_value):
        shape = jnp.shape(field_value)
        dtype = jnp.result_type(field_value)
        return jnp.zeros((B,) + shape, dtype=dtype)

    return type(dataclass_instance)(**{
        f.name: promote(getattr(dataclass_instance, f.name))
        for f in fields(dataclass_instance)
    })


def insert_dataclass_instance_at_index(
    batched_instance: struct.dataclass,
    single_instance: struct.dataclass,
    index: int
) -> struct.dataclass:
    """
    Inserts a single dataclass instance into a batched dataclass at a specified index.

    Parameters:
        batched_instance: A struct.dataclass with array fields of shape (B, ...).
        single_instance: A struct.dataclass of the same type, with unbatched fields of shape (...).
        index: Integer index into the batch dimension where the single_instance should be stored.

    Returns:
        A new struct.dataclass instance with the updated fields at the specified index.
        Other indices remain unchanged.
    """
    return type(batched_instance)(**{
        f.name: getattr(batched_instance, f.name).at[index].set(getattr(single_instance, f.name))
        for f in fields(batched_instance)
    })


def update_dataclass_at_index(
    batched_instance: struct.dataclass,
    single_instance: struct.dataclass,
    index: int
) -> struct.dataclass:
    """
    Returns a new struct.dataclass where the fields at the given `index`
    are updated with values from `single_instance`.

    This version performs an efficient in-place-style update using .at[].set(),
    avoiding full reconstruction of the dataclass.

    Parameters:
        batched_instance: Batched struct.dataclass with array fields of shape (B, ...).
        single_instance: Unbatched instance of the same dataclass.
        index: Integer index to update.

    Returns:
        Updated batched struct.dataclass with the value inserted at `index`.
    """
    for f in fields(batched_instance):
        field_name = f.name
        batched_field = getattr(batched_instance, field_name)
        value = getattr(single_instance, field_name)
        batched_instance = batched_instance.replace(
            **{field_name: batched_field.at[index].set(value)}
        )
    return batched_instance


def truncate_dataclass(batched_instance: struct.dataclass, b: int) -> struct.dataclass:
    """
    Truncates each array field in a struct.dataclass to the first `b` elements along axis 0.
    Supports dynamic slicing in JAX-transformed contexts (e.g., vmap, jit).

    Parameters:
        x: struct.dataclass with array fields of shape (T, ...).
        b: dynamic integer (can be a Tracer) indicating how many leading elements to keep.

    Returns:
        A new struct.dataclass with each field truncated to x.field[:b] using dynamic_slice.
    """
    result = {}
    for f in fields(batched_instance):
        arr = getattr(batched_instance, f.name)
        shape = arr.shape
        # dynamic_slice needs a slice size for each dimension
        slice_sizes = (b,) + shape[1:]
        result[f.name] = lax.dynamic_slice(arr, (0,) * arr.ndim, slice_sizes)
    return type(batched_instance)(**result)


def mean_across_batch(batched_instance: struct.dataclass) -> struct.dataclass:
    """
    Computes the mean across the leading (batch) dimension for each array field
    in a struct.dataclass.

    Assumes all fields have shape (B, ...) and performs `jnp.mean(field, axis=0)`.

    Parameters:
        batched_instance: A struct.dataclass where each field is a JAX array with a batch
                          dimension as the first axis.

    Returns:
        A new struct.dataclass instance of the same type, where each field has the batch
        dimension averaged out — resulting in shape (...).
    """
    return type(batched_instance)(**{
        f.name: jnp.mean(getattr(batched_instance, f.name), axis=0)
        for f in fields(batched_instance)
    })
