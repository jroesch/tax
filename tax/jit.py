# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

An implementation of `jax.jit` which uses TVM instead of XLA, heavily
inspired by the existing implementation of `jit` in api.py.
"""

import collections
import functools
from functools import partial
import inspect
import itertools as it
from typing import (Any, Callable, Generator, Iterable, NamedTuple, Mapping,
                    Optional, Sequence, Tuple, TypeVar, Union, overload, Dict,
                    Hashable, List)
from typing_extensions import Literal
from warnings import warn

import numpy as np
from contextlib import contextmanager, ExitStack

import jax
from jax import core
from jax import linear_util as lu
from jax import stages
from jax.core import eval_jaxpr
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           tree_structure, tree_transpose, tree_leaves,
                           tree_multimap, treedef_is_leaf, treedef_children,
                           Partial, PyTreeDef, all_leaves, treedef_tuple)

from jax._src import device_array
from jax._src import dispatch
from jax._src import dtypes
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src.api_util import (
    flatten_fun, apply_flat_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2,
    argnums_partial, argnums_partial_except, flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    shaped_abstractify, _ensure_str_tuple, argnames_partial_except, validate_argnames, validate_argnums)
from jax._src.lax import lax as lax_internal
from jax._src.lib import jax_jit
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.lib import pmap_lib
from jax._src.lib import xla_extension_version
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import broadcast_prefix
from jax._src.util import (unzip2, curry, safe_map, safe_zip, prod, split_list,
                           extend_name_stack, new_name_stack, wrap_name, cache,
                           wraps, HashableFunction)

# Unused imports to be exported
from jax._src.lib.xla_bridge import (device_count, local_device_count, devices,
                                     local_devices, process_index,
                                     process_count, host_id, host_ids,
                                     host_count, default_backend)
from jax.ad_checkpoint import checkpoint_policies
from jax.core import ShapedArray, raise_to_shaped
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import (closure_convert, custom_gradient, custom_jvp,
                                    custom_vjp, linear_call)
from jax.custom_transpose import custom_transpose
from jax.interpreters import partial_eval as pe
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters import masking

from jax._src.config import (
    flags, config, bool_env,
    disable_jit as _disable_jit,
    debug_nans as config_debug_nans,
    debug_infs as config_debug_infs,
    _thread_local_state as config_thread_local_state,
    explicit_device_put_scope as config_explicit_device_put_scope,
    explicit_device_get_scope as config_explicit_device_get_scope)

from tax.lowering import tvm_lower_jaxpr, tvm_exec


# traceback_util.register_exclusion(__file__)

_dtype = partial(dtypes.dtype, canonicalize=True)

AxisName = Any

# These TypeVars are used below to express the fact that function types
# (i.e. call signatures) are invariant under the vmap transformation.
F = TypeVar("F", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

def _nan_check_posthook(fun, args, kwargs, output):
  """Hook function called by the C++ jit/pmap to perform NaN checking."""
  leaves = tree_leaves(output)

  buffers = []
  for da_or_sda in leaves:
    if hasattr(da_or_sda, "device_buffer"):
      buffers.append(da_or_sda.device_buffer)
    elif hasattr(da_or_sda, "device_buffers"):
      buffers.extend(da_or_sda.device_buffers)

  try:
    dispatch.check_special(xla.xla_call_p, buffers)
  except FloatingPointError:
    # compiled_fun can only raise in this case
    assert config.jax_debug_nans or config.jax_debug_infs
    print("Invalid nan value encountered in the output of a C++-jit/pmap "
          "function. Calling the de-optimized version.")
    fun._cache_miss(*args, **kwargs)[0]  # probably won't return

def _update_debug_special_global(_):
  if config._read("jax_debug_nans") or config._read("jax_debug_infs"):
    jax_jit.global_state().post_hook = _nan_check_posthook
  else:
    jax_jit.global_state().post_hook = None

def _update_debug_special_thread_local(_):
  if (getattr(config_thread_local_state, "jax_debug_nans", False) or
      getattr(config_thread_local_state, "jax_debug_infs", False)):
    jax_jit.thread_local_state().post_hook = _nan_check_posthook
  else:
    jax_jit.thread_local_state().post_hook = None

config_debug_nans._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)
config_debug_infs._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)


float0 = dtypes.float0

def _check_callable(fun):
  # In Python 3.10+, the only thing stopping us from supporting staticmethods
  # is that we can't take weak references to them, which the C++ JIT requires.
  if isinstance(fun, staticmethod):
    raise TypeError(f"staticmethod arguments are not supported, got {fun}")
  if not callable(fun):
    raise TypeError(f"Expected a callable value, got {fun}")
  if _isgeneratorfunction(fun):
    raise TypeError(f"Expected a function, got a generator function: {fun}")

def _isgeneratorfunction(fun):
  # TODO 3.9+: remove
  # re-implemented here because of https://bugs.python.org/issue33261
  while inspect.ismethod(fun):
    fun = fun.__func__
  while isinstance(fun, functools.partial):
    fun = fun.func
  return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)

_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD

def _infer_argnums_and_argnames(
    sig: inspect.Signature,
    argnums: Union[int, Iterable[int], None],
    argnames: Union[str, Iterable[str], None],
  ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
  """Infer missing argnums and argnames for a function with inspect."""
  if argnums is None and argnames is None:
    return (), ()

  if argnums is not None and argnames is not None:
    argnums = _ensure_index_tuple(argnums)
    argnames = _ensure_str_tuple(argnames)

    return argnums, argnames

  parameters = sig.parameters
  if argnums is None:
    assert argnames is not None
    argnames = _ensure_str_tuple(argnames)
    argnums = tuple(
        i for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and k in argnames
    )
  else:
    argnums = _ensure_index_tuple(argnums)
    argnames = tuple(
        k for i, (k, param) in enumerate(parameters.items())
        if param.kind == _POSITIONAL_OR_KEYWORD and i in argnums
    )

  return argnums, argnames

def jit(
    fun: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    device: Optional[xc.Device] = None,
    backend: Optional[str] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    abstracted_axes: Optional[Any] = None,
  ) -> stages.Wrapped:
  """Sets up ``fun`` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. ``fun`` should be a pure function, as
      side-effects may only be executed once.

      The arguments and return value of ``fun`` should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.

      JAX keeps a weak reference to ``fun`` for use as a compilation cache key,
      so the object ``fun`` must be weakly-referenceable. Most :class:`Callable`
      objects will already satisfy this requirement.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.
      Note that donate_argnums only work for positional arguments, and keyword
      arguments will not be donated.

      For more details on buffer donation see the
      [FAQ](https://jax.readthedocs.io/en/latest/faq.html#buffer-donation).

    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

  Examples:
    In the following example, ``selu`` can be compiled into a single fused kernel
    by XLA:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def selu(x, alpha=1.67, lmbda=1.05):
    ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
    >>>
    >>> key = jax.random.PRNGKey(0)
    >>> x = jax.random.normal(key, (10,))
    >>> print(selu(x))  # doctest: +SKIP
    [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
    -0.85743 -0.78232  0.76827  0.59566 ]

    To pass arguments such as ``static_argnames`` when decorating a function, a common
    pattern is to use :func:`functools.partial`:

    >>> from functools import partial
    >>>
    >>> @partial(jax.jit, static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    DeviceArray([   0,    1,  256, 6561], dtype=int32)
  """
  # Implemements common logic between CPP and Python backends
  _check_callable(fun)

  # Coerce input
  donate_argnums = _ensure_index_tuple(donate_argnums)

  try:
    sig = inspect.signature(fun)
  except ValueError:
    # Some built-in functions don't support signature.
    # See: https://github.com/python/cpython/issues/73485
    # In this case no validation is done
    static_argnums = () if static_argnums is None else _ensure_index_tuple(static_argnums)
    static_argnames = () if static_argnames is None else _ensure_str_tuple(static_argnames)
  else:
    # Infer argnums and argnames according to docstring
    static_argnums, static_argnames = _infer_argnums_and_argnames(
        sig, static_argnums, static_argnames)

    # Validation
    validate_argnums(sig, static_argnums, "static_argnums")
    validate_argnums(sig, donate_argnums, "donate_argnums")

    validate_argnames(sig, static_argnames, "static_argnames")

  # Compensate for static argnums absorbing args
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  return _python_jit(fun, static_argnums=static_argnums, static_argnames=static_argnames,
                      device=device, backend=backend, donate_argnums=donate_argnums,
                      inline=inline, keep_unused=keep_unused, abstracted_axes=abstracted_axes)

def _prepare_jit(fun, static_argnums, static_argnames, donate_argnums,
                 args, kwargs):
  # Validate donate_argnums
  if max(donate_argnums, default=-1) >= len(args):
    raise ValueError(
        f"jitted function has donate_argnums={donate_argnums} but "
        f"was called with only {len(args)} positional arguments.")

  f = lu.wrap_init(fun)
  f, args = argnums_partial_except(f, static_argnums, args, allow_invalid=True)
  f, kwargs = argnames_partial_except(f, static_argnames, kwargs)
  args_flat, in_tree = tree_flatten((args, kwargs))
  if donate_argnums:
    donated_invars = donation_vector(donate_argnums, args, kwargs)
  else:
    donated_invars = (False,) * len(args_flat)

  return f, in_tree, args_flat, donated_invars


PytreeOfAbstractedAxesSpec = Any

def arg_spec(x):
  # like xla.arg_spec but duck-types on x.shape and x.dtype
  aval = None if jax.config.jax_dynamic_shapes else shaped_abstractify(x)
  device = getattr(x, '_device', None)
  return aval, device

def _python_jit(
    fun: Callable,
    *,
    static_argnums: Tuple[int, ...],
    static_argnames: Tuple[str, ...],
    device: Optional[xc.Device],
    backend: Optional[str],
    donate_argnums: Tuple[int, ...],
    inline: bool,
    keep_unused: bool,
    abstracted_axes: Optional[PytreeOfAbstractedAxesSpec],
  ) -> stages.Wrapped:
  @wraps(fun)
  @api_boundary
  def f_jitted(*args, **kwargs):
    if config.jax_disable_jit:
      return fun(*args, **kwargs)

    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    arg_specs_and_devices = map(arg_spec, args_flat)

    for arg in args_flat:
      _check_arg(arg)

    if config.jax_dynamic_shapes:
      axes_specs = (None if abstracted_axes is None else
                    _flat_axes_specs(abstracted_axes, *args, **kwargs))
      in_type = pe.infer_lambda_input_type(axes_specs, args_flat)
      flat_fun = lu.annotate(flat_fun, in_type)


    # out_flat = xla.xla_call(
    #     flat_fun, *args_flat,
    #     device=device, backend=backend, name=flat_fun.__name__,
    #     donated_invars=donated_invars, inline=inline,
    #     keep_unused=keep_unused)

    module = tvm_lower_jaxpr(
        flat_fun, arg_specs_and_devices,
        device=device, backend=backend, name=flat_fun.__name__,
        donated_invars=donated_invars, inline=inline,
        keep_unused=keep_unused)

    out_flat = tvm_exec(module, args_flat)

    return tree_unflatten(out_tree(), out_flat)

  f_jitted.lower = _jit_lower(fun, static_argnums, static_argnames, device,
                              backend, donate_argnums, inline, keep_unused)

  def clear_cache():
    dispatch.xla_callable.evict_function(fun)
  f_jitted.clear_cache = clear_cache

  return f_jitted

def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> List[pe.AbstractedAxesSpec]:
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)


class _BackendAndDeviceInfo(NamedTuple):
  default_device: xc.Device
  committed_to_device: bool

class _FastpathData(NamedTuple):
  xla_executable: xla.XlaExecutable
  out_pytree_def: Any
  sticky_device: xc.Device
  avals: Iterable[Any]
  lazy_exprs: Iterable[Any]
  kept_var_bitvec: Iterable[bool]

def _jit_lower(fun, static_argnums, static_argnames, device, backend,
               donate_argnums, inline,  keep_unused: bool):
  """Make a ``lower`` method for jitted functions."""
  # If the function we returned from ``jit`` were a class instance,
  # this might naturally be a method, with ``fun`` as a ``self`` and
  # all the other arguments stored as attributes.

  def arg_spec(x):
    # like xla.arg_spec but duck-types on x.shape and x.dtype
    aval = shaped_abstractify(x)
    try:
      return aval, x._device
    except:
      return aval, None

  @api_boundary
  def lower(*args, **kwargs) -> stages.Lowered:
    """Lower this function for the given arguments.

    A lowered function is staged out of Python and translated to a
    compiler's input language, possibly in a backend-dependent
    manner. It is ready for compilation but not yet compiled.

    Returns:
      A ``Lowered`` instance representing the lowering.
    """
    closed_fun, in_tree, args_flat, donated_invars = _prepare_jit(
        fun, static_argnums, static_argnames, donate_argnums, args, kwargs)
    flat_fun, out_tree = flatten_fun(closed_fun, in_tree)
    name = flat_fun.__name__
    arg_specs_and_device = list(unsafe_map(arg_spec, args_flat))
    # Only do this if the list is not empty
    if arg_specs_and_device:
      arg_specs = zip(*arg_specs_and_device)[0]
    else:
      arg_specs = []
    computation = dispatch.lower_xla_callable(flat_fun, device, backend, name,
                                              donated_invars, True,
                                              keep_unused,
                                              *arg_specs_and_device)
    return stages.Lowered.from_flat_info(
        computation, in_tree, arg_specs, donate_argnums, out_tree())

  return lower

def xla_computation(fun: Callable,
                    static_argnums: Union[int, Iterable[int]] = (),
                    axis_env: Optional[Sequence[Tuple[AxisName, int]]] = None,
                    in_parts=None, out_parts=None,
                    backend: Optional[str] = None,
                    tuple_args: bool = False,
                    instantiate_const_outputs: Optional[bool] = None,
                    return_shape: bool = False,
                    donate_argnums: Union[int, Iterable[int]] = ()) -> Callable:
  """Creates a function that produces its XLA computation given example args.

  Args:
    fun: Function from which to form XLA computations.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`. See the examples below.
    in_parts: Optional, how each argument to ``fun`` should be partitioned or
      replicated. This is used to specify partitioned XLA computations, see
      ``sharded_jit`` for more info.
    out_parts: Optional, how each output of ``fun`` should be partitioned or
      replicated. This is used to specify partitioned XLA computations, see
      ``sharded_jit`` for more info.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    tuple_args: Optional bool, defaults to ``False``. If ``True``, the resulting
      XLA computation will have a single tuple argument that is unpacked into
      the specified function arguments. If `None`, tupling will be enabled when
      there are more than 100 arguments, since some platforms have limits on
      argument arity.
    instantiate_const_outputs: Deprecated argument, does nothing.
    return_shape: Optional boolean, defaults to ``False``. If ``True``, the
      wrapped function returns a pair where the first element is the XLA
      computation and the second element is a pytree with the same structure as
      the output of ``fun`` and where the leaves are objects with ``shape``,
      ``dtype``, and ``named_shape`` attributes representing the corresponding
      types of the output leaves.
    donate_argnums: Specify which arguments are "donated" to the computation.
      It is safe to donate arguments if you no longer need them once the
      computation has finished. In some cases XLA can make use of donated
      buffers to reduce the amount of memory needed to perform a computation,
      for example recycling one of your input buffers to store a result. You
      should not reuse buffers that you donate to a computation, JAX will raise
      an error if you try to.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
    a built XLA Computation (see xla_client.py), from which representations of
    the unoptimized XLA HLO computation can be extracted using methods like
    ``as_hlo_text``, ``as_serialized_hlo_module_proto``, and
    ``as_hlo_dot_graph``. If the argument ``return_shape`` is ``True``, then the
    wrapped function returns a pair where the first element is the XLA
    Computation and the second element is a pytree representing the structure,
    shapes, dtypes, and named shapes of the output of ``fun``.

    Concrete example arguments are not always necessary. For those arguments not
    indicated by ``static_argnums``, any object with ``shape`` and ``dtype``
    attributes is acceptable (excepting namedtuples, which are treated as Python
    containers).

  For example:

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> c = jax.xla_computation(f)(3.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule xla_computation_f.6
  <BLANKLINE>
  ENTRY xla_computation_f.6 {
    constant.2 = pred[] constant(false)
    parameter.1 = f32[] parameter(0)
    cosine.3 = f32[] cosine(parameter.1)
    sine.4 = f32[] sine(cosine.3)
    ROOT tuple.5 = (f32[]) tuple(sine.4)
  }
  <BLANKLINE>
  <BLANKLINE>


  Alternatively, the assignment to ``c`` above could be written:

  >>> import types
  >>> scalar = types.SimpleNamespace(shape=(), dtype=np.dtype(np.float32))
  >>> c = jax.xla_computation(f)(scalar)


  Here's an example that involves a parallel collective and axis name:

  >>> def f(x): return x - jax.lax.psum(x, 'i')
  >>> c = jax.xla_computation(f, axis_env=[('i', 4)])(2)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation.9
  primitive_computation.3 {
    parameter.4 = s32[] parameter(0)
    parameter.5 = s32[] parameter(1)
    ROOT add.6 = s32[] add(parameter.4, parameter.5)
  }
  ENTRY jaxpr_computation.9 {
    tuple.1 = () tuple()
    parameter.2 = s32[] parameter(0)
    all-reduce.7 = s32[] all-reduce(parameter.2), replica_groups={{0,1,2,3}}, to_apply=primitive_computation.3
    ROOT subtract.8 = s32[] subtract(parameter.2, all-reduce.7)
  }
  <BLANKLINE>
  <BLANKLINE>

  Notice the ``replica_groups`` that were generated. Here's an example that
  generates more interesting ``replica_groups``:

  >>> from jax import lax
  >>> def g(x):
  ...   rowsum = lax.psum(x, 'i')
  ...   colsum = lax.psum(x, 'j')
  ...   allsum = lax.psum(x, ('i', 'j'))
  ...   return rowsum, colsum, allsum
  ...
  >>> axis_env = [('i', 4), ('j', 2)]
  >>> c = xla_computation(g, axis_env=axis_env)(5.)
  >>> print(c.as_hlo_text())  # doctest: +SKIP
  HloModule jaxpr_computation__1.19
  [removed uninteresting text here]
  ENTRY jaxpr_computation__1.19 {
    tuple.1 = () tuple()
    parameter.2 = f32[] parameter(0)
    all-reduce.7 = f32[] all-reduce(parameter.2), replica_groups={{0,2,4,6},{1,3,5,7}}, to_apply=primitive_computation__1.3
    all-reduce.12 = f32[] all-reduce(parameter.2), replica_groups={{0,1},{2,3},{4,5},{6,7}}, to_apply=primitive_computation__1.8
    all-reduce.17 = f32[] all-reduce(parameter.2), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=primitive_computation__1.13
    ROOT tuple.18 = (f32[], f32[], f32[]) tuple(all-reduce.7, all-reduce.12, all-reduce.17)
  }
  """
  del instantiate_const_outputs  # Unused

  _check_callable(fun)
  static_argnums = _ensure_index_tuple(static_argnums)
  donate_argnums = _ensure_index_tuple(donate_argnums)
  donate_argnums = rebase_donate_argnums(donate_argnums, static_argnums)

  fun_name = getattr(fun, "__name__", "unknown")

  backend = backend if backend is not None else xb.get_backend().platform

  def make_axis_env(nreps):
    if axis_env is None:
      return xla.AxisEnv(nreps, (), ())
    else:
      nreps = nreps * prod(size for name, size in axis_env)
      names, sizes = unzip2(axis_env)
      return xla.AxisEnv(nreps, names, sizes)

  @wraps(fun)
  @api_boundary
  def computation_maker(*args, **kwargs):
    if max(static_argnums + donate_argnums, default=-1) >= len(args):
      raise ValueError(f"jitted function has static_argnums={static_argnums},"
                       f" donate_argnums={donate_argnums} but "
                       f"was called with only {len(args)} positional arguments.")

    f = lu.wrap_init(fun)
    f, dyn_args = argnums_partial_except(f, static_argnums, args, allow_invalid=False)
    args_flat, in_tree = tree_flatten((dyn_args, kwargs))
    if donate_argnums:
      donated_invars = donation_vector(donate_argnums, dyn_args, kwargs)
    else:
      donated_invars = (False,) * len(args_flat)

    if in_parts is None:
      in_parts_flat = None
    else:
      in_parts_flat = tuple(flatten_axes(
          "xla_computation in_parts", in_tree.children()[0], in_parts))
    jaxtree_fun, out_tree = flatten_fun(f, in_tree)
    avals = map(shaped_abstractify, args_flat)
    with ExitStack() as stack:
      for axis_name, size in axis_env or []:
        stack.enter_context(core.extend_axis_env(axis_name, size, None))
      jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(jaxtree_fun, avals)
      jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)
      axis_env_ = make_axis_env(dispatch.jaxpr_replicas(jaxpr))
      if out_parts is None:
        out_parts_flat = None
      else:
        out_parts_flat = tuple(flatten_axes(
            "xla_computation out_parts", out_tree(), out_parts))
      unordered_effects = [eff for eff in jaxpr.effects
                           if eff not in core.ordered_effects]
      ordered_effects = [eff for eff in jaxpr.effects
                         if eff in core.ordered_effects]
      m, _ = mlir.lower_jaxpr_to_module(
          f"xla_computation_{fun_name}",
          core.ClosedJaxpr(jaxpr, consts),
          unordered_effects=unordered_effects,
          ordered_effects=ordered_effects,
          platform=backend,
          axis_context=mlir.ReplicaAxisContext(axis_env_),
          name_stack=new_name_stack(wrap_name(fun_name, "xla_computation")),
          donated_args=donated_invars,
          arg_shardings=(None if in_parts_flat is None else
                         map(xla.sharding_to_proto, in_parts_flat)),
          result_shardings=(None if out_parts_flat is None else
                            map(xla.sharding_to_proto, out_parts_flat)))
      should_tuple = tuple_args if tuple_args is not None else (len(avals) > 100)
      built = xc._xla.mlir.mlir_module_to_xla_computation(
          mlir.module_to_string(m), use_tuple_args=should_tuple,
          return_tuple=True)
    out_shapes_flat = [
        ShapeDtypeStruct(a.shape, a.dtype, a.named_shape) for a in out_avals]
    out_shape = tree_unflatten(out_tree(), out_shapes_flat)
    for out_aval in out_avals:
      if not isinstance(out_aval, xla.ShapedArray):
        raise RuntimeError("As we want to propagate the weak_type, we need "
                           "to get a ShapedArray, otherwise this "
                           "information is lost")

    if return_shape:
      return built, out_shape
    else:
      return built

  return computation_maker

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.get_aval(x)
  except TypeError as e:
    raise TypeError(msg(f"was {x}")) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        raise TypeError(msg(f"had shape: {aval.shape}"))
    else:
      raise TypeError(msg(f"had abstract value {aval}"))

def _possible_downcast(x, example):
  if (dtypes.issubdtype(x.dtype, np.complexfloating) and
      not dtypes.issubdtype(_dtype(example), np.complexfloating)):
    x = x.real
  dtype = None if example is None else _dtype(example)
  weak_type = None if example is None else dtypes.is_weakly_typed(example)
  return lax_internal._convert_element_type(x, dtype, weak_type)

def _unravel_array_into_pytree(pytree, axis, example, arr):
  """Unravel an array into a PyTree with a given structure.
  Args:
      pytree: The pytree that provides the structure.
      axis: The parameter axis is either -1, 0, or 1.  It controls the
        resulting shapes.
      example: If specified, cast the components to the matching dtype/weak_type,
        or else use the pytree leaf type if example is None.
      arr: The array to be unraveled.
  """
  leaves, treedef = tree_flatten(pytree)
  axis = axis % arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, np.cumsum(map(np.size, leaves[:-1])), axis)
  reshaped_parts = [
      _possible_downcast(np.reshape(x, shape), leaf if example is None else example)
      for x, shape, leaf in zip(parts, shapes, leaves)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, np.ndarray):
    return np.split(x, indices, axis)
  else:
    return x.split(indices, axis)

@curry
def shapecheck(in_shapes, out_shape, fun: Callable):
  _check_callable(fun)
  in_shapes, in_tree = tree_flatten(in_shapes)
  in_shapes = map(masking.parse_spec, in_shapes)
  out_specs, out_spec_tree = tree_flatten(out_shape)
  out_specs = map(masking.parse_spec, out_specs)
  flat_fun, out_tree_thunk = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  avals = map(partial(ShapedArray, dtype=np.float32), in_shapes)
  out_shapes = [o.shape for o in pe.abstract_eval_fun(flat_fun.call_wrapped, *avals)]
  masking.check_shapes(map(tuple, out_specs), out_spec_tree,
                       map(tuple, out_shapes), out_tree_thunk())
  return fun

def _check_arg(arg):
  if not (isinstance(arg, core.Tracer) or _valid_jaxtype(arg)):
    raise TypeError(f"Argument '{arg}' of type {type(arg)} is not a valid JAX type.")

# TODO(mattjj,necula): this duplicates code in core.valid_jaxtype, but one
# internal user relies on it for duck-typing. must fix downstream user!
def _valid_jaxtype(arg):
  try:
    xla.abstractify(arg)  # faster than core.get_aval
  except TypeError:
    return False
  else:
    return True
