import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap
from jax import random

import numpy as np
from functools import wraps

from jax import core
from jax import lax
from jax._src.util import safe_map

import tvm
from tvm import relay
from tvm import IRModule
from jax._src.ad_util import add_jaxvals_p
from jax import dlpack

from jax._src.util import (unzip2, curry, safe_map, safe_zip, prod, split_list,
                           extend_name_stack, new_name_stack, wrap_name, cache,
                           wraps, HashableFunction)

PRIM_MAP = {
    jax.lax.tanh_p: relay.op.tanh,
    jax.lax.exp_p: relay.op.exp,
    jax.lax.sub_p: relay.op.subtract,
    jax.lax.mul_p: relay.op.multiply,
    # Not sure this is actually right, but add_any seems under documented, looks like abstract addition for grad?
    # how to eval, maybe look at HLO lowering rules?
    # It looks like Lax defintions set add_any to be mapped to add for most types, see here:
    # https://github.com/google/jax/blob/847c04fd8c86f7cec69625cf7e0ade1cb80a1f72/jax/_src/lax/lax.py#L1411
    add_jaxvals_p: relay.op.add,
}

import jax.interpreters.partial_eval as pe
import jax._src.util as util
import itertools
from jax import linear_util as lu

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

# See: https://github.com/google/jax/blob/f77500c2017ca0bc9ba46bf36ee6546ad146fa7d/jax/_src/dispatch.py#L267 for more inspiration on how to do this.
def tvm_lower_jaxpr(
    flat_fun, *arg_specs_and_devices,
    device, backend, name,
    donated_invars, inline,
    keep_unused):

    fun = flat_fun
    if device is not None and backend is not None:
        raise ValueError("can't specify both a device and a backend for jit, "
                     "got device={} and backend={}".format(device, backend))

    abstract_args, arg_devices = util.unzip2(*arg_specs_and_devices)

    if fun.in_type is None:
        # Add an annotation inferred from the arguments; no dynamic axes here.
        in_type = tuple(unsafe_zip(abstract_args, itertools.repeat(True)))
        fun = lu.annotate(fun, in_type)
    else:
        assert abstract_args == (None,) * len(abstract_args)
        abstract_args = [aval for aval, _ in fun.in_type]
    # with log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "
    #                         "for jit in {elapsed_time} sec"):

    jaxpr, out_type, consts = pe.trace_to_jaxpr_final2(
        fun, pe.debug_info_final(fun, "jit"))

    out_avals, kept_outputs = util.unzip2(out_type)

    if any(isinstance(c, core.Tracer) for c in consts):
        pass # raise UnexpectedTracerError("Encountered an unexpected tracer.")

    module = IRModule()
    env = {}
    relay_vars = []
    for var in jaxpr.invars:
        if isinstance(var, core.Var):
            assert(type(var.aval) == jax.core.ShapedArray)
            rvar = relay.var(name_hint=repr(var), shape=var.aval.shape, dtype=str(var.aval.dtype))
            relay_vars.append(rvar)
            # look into why safe_map?
            env[var] = rvar
        else:
            raise Exception(f"unsupported input variable type {type(var)}")

    for eqn in jaxpr.eqns:
        assert len(eqn.params) == 0, "params not supported"
        inputs = []
        for var in eqn.invars:
            if isinstance(var, core.Var):
                inputs.append(env[var])
            elif isinstance(var, core.Literal):
                inputs.append(relay.const(var.val))
            else:
                raise Exception(f"unsupported input variable type {type(var)}")

        output = PRIM_MAP[eqn.primitive](*inputs)
        env[eqn.outvars[0]] = output

    relay_outputs = []
    for var in jaxpr.outvars:
        relay_outputs.append(env[var])

    fn = relay.Function(relay_vars, relay.Tuple(relay_outputs))
    module["main"] = fn

    return module

def tvm_exec(module, args):
    tvm_args = []
    for arg in args:
        # need a better way for formatting args for the program, look at jax.jit
        tvm_args.append(np.array(arg, dtype="float32"))

    exec = relay.create_executor("vm", device=tvm.cpu(0), target="llvm", mod=module)
    result = exec.evaluate()(*tvm_args)
    from jax import numpy as jnp
    jax_result = []
    for arr in list(result):
        # Ensure this is zero copy
        jax_array = jnp.array(arr.asnumpy())
        jax_result.append(jax_array)
    return jax_result
