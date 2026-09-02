### Sample miniprojects to get use to the PhD research ###

Shared Python modules live in [`libs/`](libs/) at the project root (MCMC, SMC, model search, etc.). Notebooks under `notebooks/` import them as `from libs import ...`. 

Activate the environment with `source activate.sh` so `PYTHONPATH` includes the repo root, or run `python setup_path.py` before importing in a notebook kernel.

Topics covered: 

1. MCMC
2. SMC Samplers 
3. Tempering Algorithms (Simulated, Parallel)
4. Machine Learning
5. Generative Models (Diffusion, etc)