import os
import numpy as np
import pandas as pd

from latentccm.causal_inf import causal_score, causal_score_direct
from latentccm import DATADIR, EXPDIR
from parameterfree import COCOB
import tensorly as tl
import math
from tqdm import tqdm

tl.set_backend("pytorch")

samples_per_sec = 100
time_bins = 10  # seconds
embed_dim = 1
time_lag = 200
num_time_series = 3
time_prop_h = 0.4
subsample_rate = 10
prop_delay_embed = 1

original_time_series = False

data_name = "Dpendulum_I"
folds = [0]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import functorch


def embed_time_series(x, lag, embed_dim):
    num_x = x.shape[0] - (embed_dim - 1) * lag
    embed_list = []
    for i in range(embed_dim):
        embed_list.append(
            x[
                (embed_dim - 1) * lag
                - (i * lag) : (embed_dim - 1) * lag
                - (i * lag)
                + num_x
            ].reshape(-1, x.shape[1])
        )
    return torch.concatenate(embed_list, axis=-1)


class MLP(nn.Module):
    def __init__(self, data_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x):
        return self.layers(x)


for fold in folds:
    print(f"Computing fold : {fold} ...")
    data_name_full = f"{data_name}_fold{fold}"
    reconstruction_name = f"{data_name}_shuffled_hidden_fold{fold}"

    df_ode_list = []
    for series in range(num_time_series):

        df_o = pd.read_csv(
            f"{DATADIR}/Dpendulum/data/{data_name}/fold_{fold}/{data_name_full}_side{series}_data.csv"
        )
        df_r = pd.read_csv(
            f"{EXPDIR}/Dpendulum/reconstructions/{reconstruction_name}_side{series}.csv"
        )
        y = np.load(
            f"{DATADIR}/Dpendulum/data/{data_name}/fold_{fold}/{data_name_full}_side{series}_full.npy"
        )

        if "random_shift" in reconstruction_name:
            random_lag = np.load(
                f"{EXPDIR}/Dependulum/reconstructions/{reconstruction_name}_side{series}_random_lag.npy"
            )
            df_r.Time = df_r.Time - np.repeat(
                random_lag, df_r.groupby("ID").size().values
            )
            df_r.drop(df_r.loc[df_r.Time < 0].index, inplace=True)
            df_r.drop(df_r.loc[df_r.Time > 10].index, inplace=True)

        n_chunks = df_o.ID.nunique() / 100

        df_r.Time = df_r.Time + df_r.ID * (n_chunks * samples_per_sec)
        df_ode0 = df_r.copy()

        if original_time_series:

            df_full = pd.DataFrame(y, columns=[f"Value_{i+1}" for i in range(4)])
            df_full["ID"] = np.repeat(np.arange(1000), 1000)[:-1]
            df_ode_list += [df_full]
        else:
            df_ode_list += [df_ode0]

    if "hidden" in reconstruction_name:
        if "shuffle" in reconstruction_name:
            x_list = []
            dx_list = []

            for df_ode in df_ode_list:
                embed_list = []
                embed_list_deriv = []

                val_c = [c for c in df_ode.columns if "Value" in c]
                deriv_c = [c for c in df_ode.columns if "Deriv" in c]

                for series_index in range(int(df_ode.ID.max())):
                    df_ = df_ode.loc[df_ode.ID == series_index, val_c]
                    df_deriv = df_ode.loc[df_ode.ID == series_index, deriv_c]

                    limit_t = int(df_.shape[0] * (1 - time_prop_h))
                    if embed_dim > 1:  # delay embedding the hiddens.
                        embed_list.append(
                            embed_time_series(
                                df_.values[limit_t:], time_lag, embed_dim
                            )[0::subsample_rate]
                        )
                        embed_list_deriv.append(
                            embed_time_series(
                                df_deriv.values[limit_t:], time_lag, embed_dim
                            )[0::subsample_rate]
                        )
                    else:
                        embed_list.append(df_.values[limit_t:][0::subsample_rate])
                        embed_list_deriv.append(
                            df_deriv.values[limit_t:][0::subsample_rate]
                        )
                x_list.append(np.concatenate(embed_list, axis=0))
                dx_list.append(np.concatenate(embed_list_deriv, axis=0))

            R = 100
            lag = 10
            embed_dim = 1

            X_tensor = torch.tensor(x_list[0]).float().cuda()
            dX_tensor = torch.tensor(dx_list[0]).float().cuda()

            avg_X_tensor = torch.mean(X_tensor, 0, keepdim=True)
            X_tensor = X_tensor - avg_X_tensor
            _, _, V = torch.linalg.svd(X_tensor, full_matrices=False)
            X_proj = X_tensor
            dX_proj = dX_tensor
            dX_tensor = dX_proj.cpu() / torch.std(X_proj, 0, keepdim=True).cpu()
            X_tensor = X_proj / torch.std(X_proj, 0, keepdim=True)
            X_tensor = embed_time_series(X_tensor, lag, embed_dim)
            dX_tensor = embed_time_series(dX_tensor, lag, embed_dim)

            Y_tensor = torch.tensor(x_list[1]).float().cuda()
            dY_tensor = torch.tensor(dx_list[1]).float().cuda()

            avg_Y_tensor = torch.mean(Y_tensor, 0, keepdim=True)
            Y_tensor = Y_tensor - avg_Y_tensor
            _, _, V = torch.linalg.svd(Y_tensor, full_matrices=False)
            Y_proj = Y_tensor
            dY_proj = dY_tensor
            dY_tensor = dY_proj.cpu() / torch.std(Y_proj, 0, keepdim=True).cpu()
            Y_tensor = Y_proj / torch.std(Y_proj, 0, keepdim=True)
            Y_tensor = embed_time_series(Y_tensor, lag, embed_dim)
            dY_tensor = embed_time_series(dY_tensor, lag, embed_dim)

            Z_tensor = torch.tensor(x_list[2]).float().cuda()
            dZ_tensor = torch.tensor(dx_list[2]).float().cuda()
            print("Y Tensor Mean", Y_tensor.mean(), "STDev", Y_tensor.std())

            avg_Z_tensor = torch.mean(Z_tensor, 0, keepdim=True)
            Z_tensor = Z_tensor - avg_Z_tensor
            _, _, V = torch.linalg.svd(Z_tensor, full_matrices=False)
            Z_proj = Z_tensor
            dZ_proj = dZ_tensor
            dZ_tensor = dZ_proj.cpu() / torch.std(Z_proj, 0, keepdim=True).cpu()
            Z_tensor = Z_proj / torch.std(Z_proj, 0, keepdim=True)
            Z_tensor = embed_time_series(Z_tensor, lag, embed_dim)
            dZ_tensor = embed_time_series(dZ_tensor, lag, embed_dim)

            N_samples = len(X_tensor)
            N_train = int(0.75 * N_samples)
            X_tensor_train = X_tensor[:N_train]
            X_tensor_val = X_tensor[N_train:]
            Y_tensor_train = Y_tensor[:N_train]
            Y_tensor_val = Y_tensor[N_train:]
            Z_tensor_train = Z_tensor[:N_train]
            Z_tensor_val = Z_tensor[N_train:]
            dX_tensor_val = dX_tensor[N_train:]
            dY_tensor_val = dY_tensor[N_train:]
            dZ_tensor_val = dZ_tensor[N_train:]

            dset_X_train = torch.utils.data.TensorDataset(X_tensor_train)
            dset_X_val = torch.utils.data.TensorDataset(X_tensor_val)
            dset_Y_train = torch.utils.data.TensorDataset(Y_tensor_train)
            dset_Y_val = torch.utils.data.TensorDataset(Y_tensor_val)
            dset_Z_train = torch.utils.data.TensorDataset(Z_tensor_train)
            dset_Z_val = torch.utils.data.TensorDataset(Z_tensor_val)

            dl_X = torch.utils.data.DataLoader(
                dset_X_train, batch_size=16, shuffle=False
            )
            dl_Y = torch.utils.data.DataLoader(
                dset_Y_train, batch_size=16, shuffle=False
            )
            dl_Z = torch.utils.data.DataLoader(
                dset_Z_train, batch_size=16, shuffle=False
            )
            dl_val_X = torch.utils.data.DataLoader(
                dset_X_val, batch_size=16, shuffle=False
            )
            dl_val_Y = torch.utils.data.DataLoader(
                dset_Y_val, batch_size=16, shuffle=False
            )
            dl_val_Z = torch.utils.data.DataLoader(
                dset_Z_val, batch_size=16, shuffle=False
            )

            # print("SHAPE", X_tensor.shape, "DTYPE", X_tensor.dtype)

            dls = {"X": dl_X, "Y": dl_Y, "Z": dl_Z}
            dls_val = {"X": dl_val_X, "Y": dl_val_Y, "Z": dl_val_Z}
            d = X_tensor.shape[1]
            mlps = {"X": MLP(d).cuda(), "Y": MLP(d).cuda(), "Z": MLP(d).cuda()}
            val_tensors = {"X": X_tensor_val, "Y": Y_tensor_val, "Z": Z_tensor_val}
            val_deriv_tensors = {
                "X": dX_tensor_val,
                "Y": dY_tensor_val,
                "Z": dZ_tensor_val,
            }
            max_nn_epochs = {"X": 50, "Y": 50, "Z": 50}
            lam = 1e-6

            for side in [
                ("X", "Z"),
                ("X", "Y"),
                ("Y", "X"),
                ("Z", "X"),
                ("Y", "Z"),
                ("Z", "Y"),
            ]:
                loss_criterion = torch.nn.MSELoss()
                model = mlps[side[0]]
                model.train()
                # print(f"Training MLP {side[0]}->{side[1]} with COCOB")
                optimizer = COCOB(model.parameters(), weight_decay=0.0)
                for epoch in tqdm(range(max_nn_epochs[side[0]]), leave=False):
                    train_loss = 0
                    model.train()
                    for i, b in enumerate(zip(dls[side[0]], dls[side[1]])):
                        optimizer.zero_grad()
                        y_hat = model(b[0][0])
                        loss = loss_criterion(y_hat, b[1][0])
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.detach()
                    train_loss /= i + 1
                    if (epoch % 20) == 0 or epoch == max_nn_epochs[side[0]] - 1:
                        continue  # print(f"Training loss at epoch {epoch}: {train_loss}")

                    train_loss = 0.0
                    model.eval()
                    for i, b in enumerate(zip(dls_val[side[0]], dls_val[side[1]])):
                        y_hat = model(b[0][0])
                        loss = loss_criterion(y_hat, b[1][0])
                        train_loss += loss.detach()
                    train_loss /= i + 1
                    # if (epoch%20)==0 or epoch==max_nn_epochs[side[0]]-1:
                    # print(f"Validation loss at epoch {epoch}: {train_loss}")

                model.eval()
                jac = (
                    functorch.vmap(functorch.jacrev(model))(val_tensors[side[0]])
                    .cpu()
                    .detach()
                )
                pf = torch.squeeze(jac @ val_deriv_tensors[side[0]][:, :, None])

                num_preds = val_tensors[side[0]].shape[1]
                sc1 = np.empty(num_preds)
                _y = val_tensors[side[1]].cpu().detach().numpy()
                _x = model(val_tensors[side[0]]).cpu().detach().numpy()
                for ii in range(num_preds):
                    sc1[ii] = np.corrcoef(_x[:, ii], _y[:, ii])[0, 1]

                average_corr = np.mean(sc1)

                print(side, "Corrcoef:", average_corr)
                print(
                    side,
                    "xmap pushforwad similarity:",
                    torch.mean(
                        F.cosine_similarity(val_deriv_tensors[side[1]], pf)
                    ).item(),
                )
        else:
            x_list = [
                df_ode[[c for c in df_ode.columns if "Value" in c]].values[
                    0::subsample_rate
                ]
                for df_ode in df_ode_list
            ]
    else:
        if "shuffle" in reconstruction_name:
            x_list = []
            for df_ode in df_ode_list:
                embed_list = []
                for series_index in range(int(df_ode.ID.max())):
                    df_ = df_ode.loc[df_ode.ID == series_index]
                    limit_t = int(df_.shape[0] * (1 - time_prop_h))
                    embed_list.append(
                        embed_time_series(
                            df_.Value_1.values[limit_t:], time_lag, embed_dim
                        )
                    )
                x_list.append(np.concatenate(embed_list[0::subsample_rate], axis=0))
        else:
            x_list = [df_ode.Value_1.values for df_ode in df_ode_list]

    if ("hidden" in reconstruction_name) or ("shuffle" in reconstruction_name):
        print("computing match between hiddens ...")
        sc1_gruode, sc2_gruode = causal_score_direct(x_list[0], x_list[1])
        sc1_gruode_init, sc2_gruode_init = causal_score_direct(
            x_list[0], x_list[1], init=True
        )

        if num_time_series == 3:
            sc13_gruode, sc31_gruode = causal_score_direct(x_list[0], x_list[2])
            sc23_gruode, sc32_gruode = causal_score_direct(x_list[1], x_list[2])

            sc13_gruode_init, sc31_gruode_init = causal_score_direct(
                x_list[0], x_list[2], init=True
            )
            sc23_gruode_init, sc32_gruode_init = causal_score_direct(
                x_list[1], x_list[2], init=True
            )

    else:
        sc1_gruode, sc2_gruode = causal_score(
            x_list[0],
            x_list[1],
            lag=time_lag,
            embed=embed_dim,
            sub_sample_rate=subsample_rate,
        )

        if num_time_series == 3:
            sc13_gruode, sc31_gruode = causal_score(
                x_list[0],
                x_list[2],
                lag=time_lag,
                embed=embed_dim,
                sub_sample_rate=subsample_rate,
            )
            sc23_gruode, sc32_gruode = causal_score(
                x_list[1],
                x_list[2],
                lag=time_lag,
                embed=embed_dim,
                sub_sample_rate=subsample_rate,
            )

    print(f"sc1 : {sc1_gruode} - sc2 {sc2_gruode}")
    print(f"sc1 init : {sc1_gruode_init} - sc2 init : {sc2_gruode_init}")
    if num_time_series == 3:
        print(f"sc31 : {sc31_gruode} - sc13 {sc13_gruode}")
        print(f"sc31 init : {sc31_gruode_init} - sc13 init : {sc13_gruode_init}")
        print(f"sc32 : {sc32_gruode} - sc23 {sc23_gruode}")
        print(f"sc32 init : {sc32_gruode_init} - sc23 init : {sc23_gruode_init}")

    results_path = f"{EXPDIR}/Dpendulum/results/results_ccm.csv"
    if os.path.exists(results_path):
        results_entry = pd.read_csv(results_path)

        results_entry.loc[results_entry.dataset_name == data_name, "sc1_gru_ode"] = (
            sc1_gruode
        )
        results_entry.loc[results_entry.dataset_name == data_name, "sc2_gru_ode"] = (
            sc2_gruode
        )
        results_entry.loc[
            results_entry.dataset_name == data_name, "sc1_gru_ode_init"
        ] = sc1_gruode_init
        results_entry.loc[
            results_entry.dataset_name == data_name, "sc2_gru_ode_init"
        ] = sc2_gruode_init
        if num_time_series == 3:
            results_entry.loc[
                results_entry.dataset_name == data_name, "sc13_gru_ode"
            ] = sc13_gruode
            results_entry.loc[
                results_entry.dataset_name == data_name, "sc31_gru_ode"
            ] = sc31_gruode
            results_entry.loc[
                results_entry.dataset_name == data_name, "sc23_gru_ode"
            ] = sc23_gruode
            results_entry.loc[
                results_entry.dataset_name == data_name, "sc32_gru_ode"
            ] = sc32_gruode

        results_entry.to_csv(results_path, index=False)


def compress_df(df, fact, time_bin=10):
    df["IDbis"] = (df.ID - 1) % fact
    df.Time = df.Time + time_bin * df.IDbis
    df.ID = df.ID - df.IDbis
    df.ID = df.ID.map(dict(zip(df.ID.unique(), np.arange(df.ID.nunique()))))
    df.drop("IDbis", axis=1, inplace=True)
    return df
