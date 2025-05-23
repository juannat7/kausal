import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import MSELoss
from torch.autograd import Variable

import numpy as np
from sklearn.metrics import balanced_accuracy_score

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 method="OLS"):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.

        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        @param method: fitting algorithm (currently, only "OLS" is supported).
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers

        self.device = device

        self.method = method

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        if self.method is "OLS":
            preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
            for k in range(self.order):
                coeff_net_k = self.coeff_nets[k]
                coeffs_k = coeff_net_k(inputs[:, k, :])
                coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
                if coeffs is None:
                    coeffs = torch.unsqueeze(coeffs_k, 1)
                else:
                    coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
                coeffs[:, k, :, :] = coeffs_k
                if self.method is "OLS":
                    preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        elif self.method is "BFT":
            NotImplementedError("Backfitting not implemented yet!")
        else:
            NotImplementedError("Unsupported fitting method!")

        return preds, coeffs


def run_epoch(epoch_num: int, model: nn.Module, optimiser: optim, predictors: np.ndarray, responses: np.ndarray,
              time_idx: np.ndarray, criterion: torch.nn.modules.loss, lmbd: float, gamma: float, batch_size: int,
              device: torch.device, alpha=0.5, verbose=True, train=True):
    """
    Runs one epoch through the dataset.

    @param epoch_num: number of the epoch (for bookkeeping only).
    @param model: model.
    @param optimiser: Torch optimiser.
    @param predictors: numpy array with predictor values of shape [N x K x p].
    @param responses: numpy array with response values of shape [N x p].
    @param time_idx: time indices of observations of shape [N].
    @param criterion: base loss criterion (e.g. MSE or CE).
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param batch_size: batch size.
    @param device: Torch device.
    @param alpha: alpha-parameter for the elastic-net (default: 0.5).
    @param verbose: print-outs enabled?
    @param train: training mode?
    @return: if train == False, returns generalised coefficient matrices and average losses incurred; otherwise, None.
    """
    if not train:
        coeffs_final = torch.zeros((predictors.shape[0], predictors.shape[1], predictors.shape[2],
                                    predictors.shape[2])).to(device)

    # Shuffle the data
    inds = np.arange(0, predictors.shape[0])
    if train:
        np.random.shuffle(inds)

    # Split into batches
    batch_split = np.arange(0, len(inds), batch_size)
    if len(inds) - batch_split[-1] < batch_size / 2:
        batch_split = batch_split[:-1]

    incurred_loss = 0
    incurred_base_loss = 0
    incurred_penalty = 0
    incurred_smoothness_penalty = 0
    for i in range(len(batch_split)):
        if i < len(batch_split) - 1:
            predictors_b = predictors[inds[batch_split[i]:batch_split[i + 1]], :, :]
            responses_b = responses[inds[batch_split[i]:batch_split[i + 1]], :]
            time_idx_b = time_idx[inds[batch_split[i]:batch_split[i + 1]]]
        else:
            predictors_b = predictors[inds[batch_split[i]:], :, :]
            responses_b = responses[inds[batch_split[i]:], :]
            time_idx_b = time_idx[inds[batch_split[i]:]]

        inputs = Variable(torch.tensor(predictors_b, dtype=torch.float)).float().to(device)
        targets = Variable(torch.tensor(responses_b, dtype=torch.float)).float().to(device)

        # Get the forecasts and generalised coefficients
        preds, coeffs = model(inputs=inputs)
        if not train:
            if i < len(batch_split) - 1:
                coeffs_final[inds[batch_split[i]:batch_split[i + 1]], :, :, :] = coeffs
            else:
                coeffs_final[inds[batch_split[i]:], :, :, :] = coeffs

        # Loss
        # Base loss
        base_loss = criterion(preds, targets)

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        penalty = (1 - alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                  alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

        # Smoothing penalty term
        next_time_points = time_idx_b + 1
        inputs_next = Variable(torch.tensor(predictors[np.where(np.isin(time_idx, next_time_points))[0], :, :],
                                            dtype=torch.float)).float().to(device)
        preds_next, coeffs_next = model(inputs=inputs_next)
        penalty_smooth = torch.norm(coeffs_next - coeffs[np.isin(next_time_points, time_idx), :, :, :], p=2)

        loss = base_loss + lmbd * penalty + gamma * penalty_smooth

        # Incur loss
        incurred_loss += loss.data.cpu().numpy()
        incurred_base_loss += base_loss.data.cpu().numpy()
        incurred_penalty += lmbd * penalty.data.cpu().numpy()
        incurred_smoothness_penalty += gamma * penalty_smooth.data.cpu().numpy()

        if train:
            # Make an optimisation step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    if verbose:
        print("Epoch " + str(epoch_num) + " : incurred loss " + str(incurred_loss) + "; incurred sparsity penalty " +
              str(incurred_penalty) + "; incurred smoothness penalty " + str(incurred_smoothness_penalty))

    if not train:
        return coeffs_final, incurred_loss / len(batch_split), incurred_base_loss / len(batch_split), \
               incurred_penalty / len(batch_split), incurred_smoothness_penalty / len(batch_split)
        
def construct_training_dataset(data, order):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # data: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        # TODO: just a hack, need a better solution...
        time_idx_r = time_idx_r + offset + 200 * (r >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
        offset = np.max(time_idx_r)
    return data_out, response, time_idx


def training_procedure(data, order: int, hidden_layer_size: int, end_epoch: int, batch_size: int, lmbd: float,
                       gamma: float, seed=42, num_hidden_layers=1, initial_learning_rate=0.001, beta_1=0.9,
                       beta_2=0.999, use_cuda=True, verbose=True, test_data=None):
    """
    Standard training procedure for GVAR model.

    @param data: numpy array with time series of shape [T x p].
    @param order: GVAR model order.
    @param hidden_layer_size: number of units in a hidden layer.
    @param end_epoch: number of training epochs.
    @param batch_size: batch size.
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param seed: random generator seed.
    @param num_hidden_layers: number oh hidden layers.
    @param initial_learning_rate: learning rate.
    @param use_cuda: whether to use GPU?
    @param verbose: print-outs enabled?
    @param test_data: optional test data.
    @return: returns an estimate of the GC dependency structure, generalised coefficient matrices, and the test MSE,
    if test data provided.
    """
    # Set random generator seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Check for CUDA availability
    if use_cuda and not torch.cuda.is_available():
        print("WARNING: CUDA is not available!")
        device = torch.device("cpu")
    elif use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Number of variables, p
    if isinstance(data, list):
        num_vars = data[0].shape[1]
    else:
        num_vars = data.shape[1]

    # Construct a training dataset with lagged time series values as predictors and future values as responses
    predictors, responses, time_idx = construct_training_dataset(data=data, order=order)

    if test_data is not None:
        # Construct a test set with lagged time series values as predictors and future values as responses
        predictors_test, responses_test, time_idx_test = construct_training_dataset(data=test_data, order=order)

    # Model definition
    senn = SENNGC(num_vars=num_vars, order=order, hidden_layer_size=hidden_layer_size,
                  num_hidden_layers=num_hidden_layers, device=device)
    senn.to(device=device)

    # Optimiser
    optimiser = optim.Adam(params=senn.parameters(), lr=initial_learning_rate, betas=(beta_1, beta_2))

    # Loss criterion
    criterion = MSELoss()

    # Run the training and testing
    for epoch in range(end_epoch):
        if verbose:
            print()

        # Train
        run_epoch(epoch_num=epoch, model=senn, optimiser=optimiser, predictors=predictors, responses=responses,
                  time_idx=time_idx, criterion=criterion, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                  device=device, train=True, verbose=verbose)

    # Compute generalised coefficients & estimate causal structure
    with torch.no_grad():
        coeffs, l, mse, pen1, pen2 = run_epoch(epoch_num=end_epoch, model=senn, optimiser=optimiser,
                                               predictors=predictors, responses=responses, time_idx=time_idx,
                                               criterion=criterion, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                                               device=device, train=False, verbose=verbose)
        causal_struct_estimate = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0)[0].cpu().numpy()

        if test_data is not None:
            # Make predictions on the test data
            coeffs_test, l_test, mse_test, pen1_test, pen2_test = run_epoch(epoch_num=(end_epoch+1), model=senn,
                                                optimiser=optimiser, predictors=predictors_test,
                                                responses=responses_test,  time_idx=time_idx_test, criterion=criterion,
                                                lmbd=lmbd, gamma=gamma,  batch_size=batch_size, device=device,
                                                train=False, verbose=verbose)
            # Return test MSE in addition to the inference results
            return causal_struct_estimate, coeffs.cpu().numpy(), mse_test
        else:
            return causal_struct_estimate, coeffs.cpu().numpy()


def training_procedure_trgc(data, order: int, hidden_layer_size: int, end_epoch: int, batch_size: int, lmbd: float,
                            gamma: float, seed=42, num_hidden_layers=1, initial_learning_rate=0.001, beta_1=0.9,
                            beta_2=0.999, Q=20, use_cuda=True, verbose=True, display=False, true_struct=None,
                            signed=False):
    """
    Stability-based estimation of the GC structure using GVAR model and time reversed GC (TRGC). Sparsity level is
    chosen to maximise the agreement between GC structures inferred on original and time-reversed time series.

    @param data: numpy array with time series of shape [T x p].
    @param order: GVAR model order.
    @param hidden_layer_size: number of units in a hidden layer.
    @param end_epoch: number of training epochs.
    @param batch_size: batch size.
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param seed: random generator seed.
    @param num_hidden_layers: number oh hidden layers.
    @param initial_learning_rate: learning rate.
    @param Q: number of quantiles (spaced equally) to consider for thresholding (default: 20).
    @param use_cuda:  whether to use GPU?
    @param verbose: print-outs enabled?
    @param display: plot stability across considered sparsity levels?
    @param true_struct: ground truth GC structure (for plotting stability only).
    @param signed: detect signs of GC interactions?
    @return: an estimate of the GC summary graph adjacency matrix, strengths of GC interactions, and generalised
    coefficient matrices. If signed == True, in addition, signs of GC interactions are returned.
    """
    data_1 = None
    data_2 = None
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
        data_1 = data
        data_2 = np.flip(data, axis=0)
    else:
        data_1 = data
        data_2 = np.flip(data, axis=0)

    if verbose:
        print("-" * 25)
        print("Running TRGC selection...")
        print("Training model #1...")
    a_hat_1, coeffs_full_1 = training_procedure(data=[data_1], order=order, hidden_layer_size=hidden_layer_size,
                                                end_epoch=end_epoch, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                                                seed=seed, num_hidden_layers=num_hidden_layers,
                                                initial_learning_rate=initial_learning_rate, beta_1=beta_1,
                                                beta_2=beta_2, use_cuda=use_cuda, verbose=False)
    if verbose:
        print("Training model #2...")
    a_hat_2, coeffs_full_2 = training_procedure(data=[data_2], order=order, hidden_layer_size=hidden_layer_size,
                                                end_epoch=end_epoch, lmbd=lmbd, gamma=gamma, batch_size=batch_size,
                                                seed=seed, num_hidden_layers=num_hidden_layers,
                                                initial_learning_rate=initial_learning_rate, beta_1=beta_1,
                                                beta_2=beta_2, use_cuda=use_cuda, verbose=False)
    a_hat_2 = np.transpose(a_hat_2)

    p = a_hat_1.shape[0]

    if verbose:
        print("Evaluating stability...")
    alphas = np.linspace(0, 1, Q)
    qs_1 = np.quantile(a=a_hat_1, q=alphas)
    qs_2 = np.quantile(a=a_hat_2, q=alphas)
    agreements = np.zeros((len(alphas), ))
    if true_struct is not None:
        agreements_ground = np.zeros((len(alphas), ))
    else:
        agreements_ground = None
    for i in range(len(alphas)):
        a_1_i = (a_hat_1 >= qs_1[i]) * 1.0
        a_2_i = (a_hat_2 >= qs_2[i]) * 1.0
        # NOTE: we ignore diagonal elements when evaluating stability
        agreements[i] = (balanced_accuracy_score(y_true=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                 y_pred=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten()) +
                         balanced_accuracy_score(y_pred=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                 y_true=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten())) / 2
        # If only self-causal relationships are inferred, then set agreement to 0
        if np.sum(a_1_i) <= p or np.sum(a_2_i) <= p:
            agreements[i] = 0
        # If all potential relationships are inferred, then set agreement to 0
        if np.sum(a_1_i) == p**2 or np.sum(a_2_i) == p**2:
            agreements[i] = 0
        if true_struct is not None:
            agreements_ground[i] = balanced_accuracy_score(y_true=true_struct[np.logical_not(np.eye(true_struct.shape[0]))].flatten(),
                                                           y_pred=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten())
    alpha_opt = alphas[np.argmax(agreements)]
    if display:
        plot_stability(alphas, agreements, agreements_ground=agreements_ground)
    if verbose:
        print("Max. stab. = " + str(np.round(np.max(agreements), 3)) + ", at α = " + str(alpha_opt))

    q_1 = np.quantile(a=a_hat_1, q=alpha_opt)
    q_2 = np.quantile(a=a_hat_2, q=alpha_opt)
    a_hat_binary = (a_hat_1 >= q_1) * 1.0

    if not signed:
        return a_hat_binary, a_hat_1, coeffs_full_1
    else:
        return a_hat_binary, a_hat_1, np.squeeze(np.median(coeffs_full_1, axis=0)) * a_hat_binary, coeffs_full_1
