import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

env = make_craftax_env_from_name("Craftax-Symbolic-v1", True)
env_params = env.default_params

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
obs, env_state = env.reset(_rng, env_params)

print(jnp.max(obs), jnp.min(obs))