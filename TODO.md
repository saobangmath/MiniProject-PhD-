1. Normalizing Flow to simulate Half-Moon. 
 - NF with NICE and rescaling layer. [Done]


1.5. Sampling with Diffusion + NF with resampling. 
    How to determine recentering: 
    - for sample: calc mean + covariate matrix 
    - shift the dataset to (0, Id) 
    - Input: very hard to sample density -> KL(q_theta(X_0:T), pi(x_0:
    T))
    - Compare with MCMC. 

X -> K-1 (X - mean)
p(x) -> | det(K) 
[Done]

Read stuffs relate to state space model, ARMA, GARCH, Online Learning 