import torch

from .base import BaseRegressor

class DMD(BaseRegressor):
    """
    Estimate Koopman Operator with Dynamic Mode Decomposition (DMD) exact method.

    Parameters:
        svd_rank (int): rank of truncation. If None, no truncation is computed.
        tikhonov_regularization (NoneType or float): Tikhonov parameter for regularization. 
            If `None`, no regularization is applied, 
            If `float`, it is used as the tikhonov parameter.
    """

    def __init__(
        self,
        svd_rank = None,
        tikhonov_regularization = None
    ):
        super().__init__()
        self.svd_rank = svd_rank
        self.tikhonov_regularization = tikhonov_regularization

    def forward(
        self, 
        W = None, 
        Wt = None
    ):
        if W is None or Wt is None:
            raise ValueError("Observables at both current and shifted step must be provided.")

        assert W.ndim == 2 and Wt.ndim == 2, "Observables must have a dimension of (n_features, n_timestep)"

        # SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        if self.svd_rank is not None:
            U = U[..., :self.svd_rank]
            S = S[:self.svd_rank]
            Vh = Vh[:self.svd_rank, ...]

        # Post-process S (diagonalization, regularization)
        S = torch.diag(S)
        if self.tikhonov_regularization is not None:
            S = self._tikhonov_regularization(S, W_norm=torch.linalg.norm(W))
    
        # Low-rank approximation of K
        K = self._least_square_operator(U = U, S_inv = torch.reciprocal(S), Vh = Vh, Wt = Wt)
        return K

    
    def _tikhonov_regularization(self, S, W_norm):
        """
        Perform Tikhonov regularization on the singular values for added stability.

        Parameters:
            S (torch.Tensor): Singular values diagonal matrix.
            W_norm (torch.Tensor): The norm of the observable.

        Returns:
            S_reg (torch.Tensor): Regularized singular value diagnomal matrix.
        """
        return (S**2 + self.tikhonov_regularization * W_norm) * torch.reciprocal(S)


    def _least_square_operator(self, U, S_inv, Vh, Wt):
        """
        Calculates the least square estimation K.

        Parameters:
            U (torch.Tensor): Left singular vectors, shape (n_features, svd_rank).
            S_inv (torch.Tensor): Inverse of singular diagonal matrix.
            Vh (torch.Tensor): Right singular vectors, shape (svd_rank, n_features).
            Wt (torch.Tensor): Prediction observables, shape (n_features, n_timestep).

        Returns:
            K (torch.Tensor): Estimated Koopman matrix in the full observable space.
        """
        return torch.linalg.multi_dot([
            Wt, 
            Vh.T.conj(), 
            S_inv, 
            U.T.conj()
        ]) 
        