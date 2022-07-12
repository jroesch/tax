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

from .jit import jit

from jax.config import config

config.experimental_cpp_jit = False

def example(x):
  return jnp.exp(jnp.tanh(x))

print(jit(example)(1.0))
