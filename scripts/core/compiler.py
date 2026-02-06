from __future__ import annotations

from typing import Any, Tuple


def _jax_array_types() -> Tuple[type, ...]:
    types: Tuple[type, ...] = ()
    try:
        from jax import Array as _JaxArray  # type: ignore
        types = types + (_JaxArray,)
    except Exception:
        pass
    try:
        from jaxlib.xla_extension import ArrayImpl as _ArrayImpl  # type: ignore
        types = types + (_ArrayImpl,)
    except Exception:
        pass
    return types


def assert_jax_array(name: str, x: Any, dtype: Any) -> None:
    types = _jax_array_types()
    if types and not isinstance(x, types):
        raise TypeError(f"{name} must be a JAX array, got {type(x)}")
    if getattr(x, "dtype", None) != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {getattr(x, 'dtype', None)}")
