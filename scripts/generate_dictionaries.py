import h5py
import spams
import numpy as np
import pandas as pd
import argschema as ags
import logging

class GenerateDictionariesParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()
    min_k = ags.fields.Integer()
    max_k = ags.fields.Integer()
    n_replicates = ags.fields.Integer(default=100)
    lambda_value = ags.fields.Float(default=0)
    gamma_value = ags.fields.Float(default=0)


def dict_learn_init(X, K):
    rng = np.random.default_rng()
    return rng.choice(X, K, axis=1, replace=False)


def generate_dictionary_replicates(data, n_components_list, lambda_param=0., gamma1_param=0., n_replicates=50):
    # x is p x n (p: number of features; n: number of samples)

    dict_results = {}
    for i, k in enumerate(n_components_list):
        # set up storage of results
        logging.info(f"k = {k}")

        dict_results[k] = np.zeros((data.shape[0], k, n_replicates))

        param = {
            "mode": 2,
            "K": k,
            "lambda1": lambda_param,
            "numThreads": -1,
            "batchsize": min(1024, data.shape[0]),
            "posD": True,
            "iter": 500,
            "modeD": 0,
            "verbose": False,
            "posAlpha": True,
            "gamma1": gamma1_param,
        }


        for j in range(n_replicates):
            if j % 10 == 0:
                logging.info(f"replicate {j}")
            param["D"] = np.asfortranarray(dict_learn_init(data, k))
            D = spams.trainDL(data, **param)
            dict_results[k][:, :, j] = D

    return dict_results


def estimate_stability(replicates_dict):
    est_stability = {}
    est_std = {}

    for k in replicates_dict:
        data = replicates_dict[k]
        n_replicates = data.shape[-1]

        dist_mat = np.zeros((n_replicates, n_replicates))

        for i in range(n_replicates):
            for j in range(i, n_replicates):
                corr = np.corrcoef(data[:, :, i], data[:, :, j], rowvar=False)
                corr = np.abs(corr)
                n_i = data[:, :, i].shape[1]
                n_j = data[:, :, j].shape[1]
                corr = corr[n_i:, :n_j]
                dist_mat[i, j] = amari_max_error(corr)
                dist_mat[j, i] = dist_mat[i, j]
        est_stability[k] = dist_mat.sum() / (n_replicates * (n_replicates - 1))
        est_std[k] = (np.sum(dist_mat ** 2) / (n_replicates * (n_replicates - 1)) -
            est_stability[k] ** 2) ** 0.5 * (2 / dist_mat.shape[0]) ** 0.5

    return est_stability, est_std



def amari_max_error(A):
    n, m = A.shape
    if n == 1 and m == 1:
        return (A == 0).sum()

    max_col = A.max(axis=1)
    col_temp = (1 - max_col).mean()

    max_row = A.max(axis=0)
    row_temp = (1 - max_row).mean()

    return (col_temp + row_temp) / 2


def main(args):
    # Load the data from an HDF5 file

    f = h5py.File(args['input_file'], 'r')
    X_input = f['data'][...]
    f.close()
    X_input = np.asfortranarray(X_input)
    logging.info(f"Generating dictionaries for {X_input.shape} size input (features by samples)")

    components_to_try = list(range(args['min_k'], args['max_k'] + 1))
    logging.info(f"Using number of components ranging from {args['min_k']} to {args['max_k']}")

    n_replicates = args['n_replicates']
    logging.info(f"Repeating dictionary generation {n_replicates} times per component size")

    replicates = generate_dictionary_replicates(
        X_input,
        components_to_try,
        n_replicates=n_replicates,
        lambda_param=args['lambda_value'],
        gamma1_param=args['gamma_value'],
        )

    logging.info("Finished generating dictionaries")

    logging.info("Estimating stability of each component size")
    stability_estimates, stability_std = estimate_stability(replicates)

    logging.info("Saving results")
    f = h5py.File(args['output_file'], 'w')
    g = f.create_group("dictionaries")
    for k in replicates:
        g.create_dataset(str(k), data=replicates[k])
    g = f.create_group("stability")
    g.create_dataset("components", data=np.array(list(stability_estimates.keys())))
    g.create_dataset("stability_values", data=np.array(list(stability_estimates.values())))
    g.create_dataset("stability_std", data=np.array(list(stability_std.values())))
    f.close()


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=GenerateDictionariesParameters)
    main(module.args)

