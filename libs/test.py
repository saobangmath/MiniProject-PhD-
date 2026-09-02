from jax import numpy as jnp 
from jax import random 

epsilon = 0.07

def test_samples(samples, target_mean, target_cov):
    r"""
     our test will demand some parameters: 
     samples (n * d)
     
    """
    N, d = samples.shape 
    assert target_cov.shape == (d, d)
    assert target_mean.shape == (d, )
    
    x = random.uniform(key = random.key(10), shape = (d, ))
    assert x.shape == (d, )

    # convert n d-dimension data to 1-dimension data 
    target_mean_1d = jnp.dot(target_mean, x)
    target_var_1d = x.T @ target_cov @ x 
    samples_1d = samples @ x
    std_var_1d = jnp.sqrt(target_var_1d)

    # get residual 
    mean_residual = abs(samples_1d.mean() - target_mean_1d) / std_var_1d
    var_residual = abs(samples_1d.var() - target_var_1d) / target_var_1d

    # chose a relevant epsilon after some observation running with multiple data 
    print(r"mean residual: {}, var residual = {}".format(mean_residual, var_residual))
    assert mean_residual <= epsilon 
    assert var_residual <= epsilon

    return target_mean_1d, target_var_1d, samples_1d

if __name__ == "__main__":
    pass 