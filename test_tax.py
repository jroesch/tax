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

def example(x):
  return jnp.exp(jnp.tanh(x))

def test_exp_tanh():
    # Example of using tax.jit
    print(jit(example)(1.0))

    # Example of using tax.jit with
    print(jit(grad(example))(1.0))
