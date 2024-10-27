import jax.numpy as jnp
from jax import jit
from jax import lax
import matplotlib.pyplot as plt
config = {"EPSILON_START":1.0,
          "EPSILON_FINISH": 0.05,
          "EPSILON_ANNEAL_TIME": 25e4}

e_values = []
for t in range(800000):
    eps = jnp.clip(  # get epsilon
                    (
                        (config["EPSILON_FINISH"] - config["EPSILON_START"])
                        / config["EPSILON_ANNEAL_TIME"]
                    )
                    * t
                    + config["EPSILON_START"],
                    config["EPSILON_FINISH"],
                )
    e_values.append(float(eps))

#plt.plot(range(800000), e_values)
#plt.show()
print("check")

start = 1.0
end = 0.05
end_fraction=0.1
e_values = []
total_steps = 800000

for t in range(800000):
    progress_remaining = (total_steps-t)/total_steps
    if (1 - progress_remaining) > end_fraction:
        eps = end
    else:
        eps = start + (1 - progress_remaining) * (end - start) / end_fraction
    e_values.append(float(eps))

plt.clf()
plt.plot(range(800000), e_values)
plt.savefig("temp.png")
plt.show()
print("check")


