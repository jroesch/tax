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

def example(x):
  return jnp.exp(jnp.tanh(x))

def jit(fun):
  @wraps(fun)
  def wrapped(*args, **kwargs):
    # Since we assume unary functions, we won't worry about flattening and
    # unflattening arguments.
    closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
    out = list(tvm_interpret_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args))
    # loosen this constraint, ie. jax flat/unflat
    return out[0]

  return wrapped

PRIM_MAP = {
    jax.lax.tanh_p: relay.op.tanh,
    jax.lax.exp_p: relay.op.exp,
}

def tvm_interpret_jaxpr(jaxpr, consts, *args):
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
        inputs = [env[var] for var in eqn.invars]
        output = PRIM_MAP[eqn.primitive](*inputs)
        env[eqn.outvars[0]] = output

    relay_outputs = []
    for var in jaxpr.outvars:
        relay_outputs.append(env[var])

    fn = relay.Function(relay_vars, relay.Tuple(relay_outputs))
    module["main"] = fn

    tvm_args = []
    for arg in args:
        # need a better way for formatting args for the program, look at jax.jit
        tvm_args.append(np.array(arg, dtype="float32"))

    exec = relay.create_executor("vm", device=tvm.cpu(0), target="llvm", mod=module)
    result = exec.evaluate()(*tvm_args)
    return result

print(jit(example)(1.0))
