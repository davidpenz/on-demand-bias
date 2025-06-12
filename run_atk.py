# %%
from src.config.config_atk import (
    configure_experiment,
    run_atk_experiments,
    run_atk_train_test,
)

if __name__ == "__main__":

    params, max_jobs = configure_experiment()

    training_fn = run_atk_train_test

    run_atk_experiments(params, max_jobs, training_fn)
# %%
# from src.utils.input_validation import find_rec_models

# exp = "/media/gustavo/Storage/dummy_debiasrec/ml-1m/RegMultVAE--2024-11-11_17-15-04"
# find_rec_models(exp)

# %%
