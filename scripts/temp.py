import jax.numpy as jnp
from jax import jit
from jax import lax
# Example arrays
A = jnp.array([[ 0.,  1.,  3.],
 [ 3.,  1.,  4.],
 [ 4.,  1.,  5.],
 [ 5. , 0. , 6.],
 [ 6. , 0.,  7.],
 [ 7.,  1. , 8.],
 [ 8. , 0. , 9.],
 [ 9.,  2., 10.]])

B = jnp.array([4, 10])


@jit
def find_matching_row(A, B):
    # Step 1: Slice A to get the first two columns
    A_first_two_columns = A[:, :2]

    # Step 2: Compare the first two columns of A with B
    matching_rows = jnp.all(A_first_two_columns == B, axis=1)

    # Step 3: Find the index of the matching row
    # Use jnp.argmax to get the first index where there's a True (assumes one match)
    matching_row_index = lax.cond(
        jnp.any(matching_rows),
        lambda _: jnp.argmax(matching_rows),  # If True, return the index of the matching row
        lambda _: 999,  # If False, return -1 to indicate no match
        operand=None  # The operand is not needed in this case, but is required by lax.cond
    )

    return matching_row_index


# Call the function
matching_row_index = find_matching_row(A, B)
print(matching_row_index)
reward = jnp.where(matching_row_index != 999, 1.0, 0.0)

random =jnp.zeros((10,1))
random = random.at[999].set(1)
print(reward)
print(random)

