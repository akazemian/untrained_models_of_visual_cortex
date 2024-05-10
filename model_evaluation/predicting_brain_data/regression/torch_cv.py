from scipy import sparse
import torch
import numpy as np
import cupy as cp
from sklearn.linear_model._ridge import  MultiOutputMixin, RegressorMixin
from sklearn.linear_model._ridge import _RidgeGCV, _BaseRidgeCV
from sklearn.linear_model._ridge import is_classifier, _check_gcv_mode
from sklearn.linear_model._ridge import _IdentityRegressor, safe_sparse_dot
from sklearn.linear_model._base import _preprocess_data
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from torchmetrics.functional import concordance_corrcoef, explained_variance
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr

def unify_dtypes(*args, target_dtype=None, precision='lowest'):
    dtypes_order = [torch.float16, torch.float32, torch.float64]
    dtypes = [arg.dtype for arg in args if isinstance(arg, torch.Tensor)]
    if not dtypes:
        return args
    if not target_dtype:
        if precision == 'highest':
            target_dtype = max(dtypes, key=dtypes_order.index)
        elif precision == 'lowest':
            target_dtype = min(dtypes, key=dtypes_order.index)
    result = tuple(arg.clone().to(dtype=target_dtype) 
                   if isinstance(arg, torch.Tensor) else arg for arg in args)
    return result[0] if len(result) == 1 else result


def convert_tensor_backend(data, backend='torch'):
    if backend == 'numpy':
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return cp.asnumpy(data) #assumes cupy
    elif backend == 'cupy':
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return data
        elif isinstance(data, np.ndarray):
            return cp.asarray(data)
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(data))
    elif backend == 'torch':
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return torch.utils.dlpack.from_dlpack(data.toDlpack())
    raise ValueError('data type and backend must be one of (numpy, cupy, torch)')
    
    
def convert_to_tensor(*args, dtype=None, device=None, copy=False):
    def convert_item(arg):
        def process_tensor(tensor):
            tensor = tensor.clone() if copy else tensor
            if dtype is not None:
                tensor = tensor.to(dtype)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        if isinstance(arg, torch.Tensor):
            return process_tensor(arg)
        elif isinstance(arg, (np.ndarray, cp.ndarray)):
            arg = convert_tensor_backend(arg, 'torch')
            return process_tensor(arg)
        elif isinstance(arg, list):
            return [convert_item(item) for item in arg]
        elif isinstance(arg, dict):
            return {key: convert_item(val) for key, val in arg.items()}
        return arg

    outputs = [convert_item(arg) for arg in args]
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')

def pearson_r_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    return(pearsonr_vec(y_true_, y_pred_)[0])



# score functions from torchmetrics.functional
_score_functions = {'spearmanr': spearman_corrcoef,
                    'pearsonr': pearson_corrcoef,
                    'concordance': concordance_corrcoef,
                    'explained_variance': explained_variance}

def get_scorer(score_type):
    return _score_functions[score_type]


class TorchEstimator:
    def __init__(self, dtype = None, device='cuda'):
        self.dtype = dtype
        self.device = device
        
    def __repr__(self):
        return 'TorchEstimator Class'

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__setattr__(attr, value.to(device))
                
    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')
        
    def remove_from_gpu(self):
        self.to('cpu')
    
    def parse_input_data(self, *args, copy=False):
        args = convert_to_tensor(*args, copy=copy, device=self.device)
        if isinstance(args, torch.Tensor):
            args = (args, )
        args = unify_dtypes(*args, target_dtype=self.dtype)
        return args
        
    def preprocess_data(self, X, y, center=[], scale=[], scaler='standard',
                        output=None, save_to_class=False, **kwargs):
    
        stats = {f'{var}_{stat}': None for stat in ['mean','std','offset','scale'] for var in ['X', 'y']}

        X, y = self.parse_input_data(X, y)
        
        def parse_preprocessing_args(*args):
            parsed_args = []
            for arg in args:
                if arg is None or len(arg) == 0:
                    parsed_args.append('none')
                elif isinstance(arg, list):
                    parsed_args.append(''.join(arg))
                else:
                    parsed_args.append(arg)
            return tuple(parsed_args)

        center, scale, output = parse_preprocessing_args(center, scale, output)
        
        if kwargs.get('fit_intercept', False):
            center += 'x'

        if 'x' in center.lower():
            stats['X_mean'] = X.mean(dim = 0)
        if 'y' in center.lower():
            stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
            
        if 'x' in scale.lower():
            stats['X_std'] = X.std(dim=0, correction=1)
            stats['X_std'][stats['X_std'] == 0.0] = 1.0  
        if 'y' in scale.lower():
            stats['y_std'] = y.std(dim=0, correction=1)
            stats['y_std'][stats['y_std'] == 0.0] = 1.0 
        
        if 'x' in center.lower():
            X -= stats['X_mean']
        if 'y' in center.lower():
            y -= stats['y_mean']
            
        if 'x' in scale.lower():
            X /= stats['X_std']
        if 'y' in scale.lower():
            y /= stats['y_std']

        if output == 'mean_std':
            if stats['X_mean'] is None:
                stats['X_mean'] = X.mean(dim=0)
            if stats['y_mean'] is None:
                stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
            if stats['X_std'] is None:
                stats['X_std'] = torch.ones(X.shape[1], dtype=X.dtype,  device=X.device)
            if stats['y_std'] is None:
                stats['y_std'] = torch.ones(y.shape[1], dtype=y.dtype,  device=y.device)
                
        if output == 'offset_scale':
            stats['X_offset'] = stats.pop('X_mean', None)
            stats['y_offset'] = stats.pop('y_mean', None)
            stats['X_scale'] = stats.pop('X_std', None)
            if stats['X_offset'] is None:
                stats['X_offset'] = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            if stats['y_offset'] is None:
                stats['y_offset'] = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
            if stats['X_scale'] is None:
                stats['X_scale'] = torch.ones(X.shape[1], dtype=X.dtype,  device=X.device)

        if save_to_class:
            for stat, value in stats.items():
                if value is not None:
                    setattr(self, stat, value)

            return X, y

        if not save_to_class:
            if output == 'offset_scale':
                return X, y, stats['X_offset'], stats['y_offset'], stats['X_scale']

            if output == 'mean_std':
                return X, y, stats['X_mean'], stats['y_mean'], stats['X_std'], stats['y_std']
            
            return X, y

### Encoding Models: TorchRidgeGCV ---------------------------------------------------------
      
class TorchRidgeGCV(TorchEstimator):
    """Ridge regression with built-in Leave-one-out Cross-Validation. """
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scale_X=False,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=False,
        device = 'cpu'
    ):
        super().__init__()
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scale_X = scale_X
        self.scoring = scoring
        self.device = device
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
            
        if isinstance(scoring, str):
            self.scorer = get_scorer(self.scoring)
        
    def __repr__(self):
        if not self.is_fitted:
            return 'TorchRidgeGCV (No Fit)' 
        if self.is_fitted:
            return 'TorchRidgeGCV (Fitted)'
        
    @staticmethod
    def _decomp_diag(v_prime, Q):
        return (v_prime * Q**2).sum(axis=-1)
    
    @staticmethod
    def _diag_dot(D, B):
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B
    
    @staticmethod
    def _find_smallest_angle(query, vectors):
        abs_cosine = torch.abs(torch.matmul(query, vectors))
        return torch.argmax(abs_cosine).item()
    
    def _compute_gram(self, X, sqrt_sw):
        X_mean = torch.zeros(X.shape[1], dtype=X.dtype, device = X.device)
        return X.matmul(X.T), X_mean
    
    def _eigen_decompose_gram(self, X, y, sqrt_sw):
        K, X_mean = self._compute_gram(X, sqrt_sw)
        if self.fit_intercept:
            K += torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)
        QT_y = torch.matmul(Q.T, y)
        return X_mean, eigvals, Q, QT_y
    
    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / torch.linalg.norm(sqrt_sw)
            intercept_dim = self._find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0  # cancel regularization for the intercept

        c = torch.matmul(Q, self._diag_dot(w, QT_y))
        G_inverse_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c
    
    def fit(self, X, y):
        self.alphas = torch.as_tensor(self.alphas, dtype=torch.float32)
        
        preprocessing_kwargs = {'output': 'offset_scale'}
        if self.fit_intercept:
            preprocessing_kwargs['center'] = 'x'
        if self.scale_X:
            preprocessing_kwargs['scale'] = 'x'

        X, y, X_offset, y_offset, X_scale = self.preprocess_data(X, y, **preprocessing_kwargs)

        decompose = self._eigen_decompose_gram
        solve = self._solve_eigen_gram

        sqrt_sw = torch.ones(X.shape[0], dtype=X.dtype, device = X.device)

        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if self.alphas.ndim == 0 else len(self.alphas)

        if self.store_cv_values:
            self.cv_values_ = torch.empty((*y.shape, n_alphas), dtype=X.dtype, device=X.device)

        best_alpha, best_coef, best_score, best_y_pred = [None]*4

        for i, alpha in enumerate(torch.atleast_1d(self.alphas)):
            G_inverse_diag, coef = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            y_pred = y - (coef / G_inverse_diag)
            if self.store_cv_values:
                self.cv_values_[:,:,i] = y_pred

            score = self.scorer(y, y_pred)
            if not self.alpha_per_target:
                score = self.scorer(y, y_pred).mean()

            # Keep track of the best model
            if best_score is None: 
                best_alpha = alpha
                best_coef = coef
                best_score = score
                best_y_pred = y_pred
                if self.alpha_per_target and n_y > 1:
                    best_alpha = torch.full((n_y,), alpha)
                    
            else: 
                if self.alpha_per_target and n_y > 1:
                    to_update = score > best_score
                    best_alpha[to_update] = alpha
                    best_coef[:, to_update] = coef[:, to_update]
                    best_score[to_update] = score[to_update]
                    best_y_pred[:, to_update] = y_pred[:, to_update]
                    
                elif score > best_score:
                    best_alpha, best_coef, best_score, best_y_pred = alpha, coef, score, y_pred

        self.alpha_ = best_alpha
        self.score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = self.dual_coef_.T.matmul(X) 
        self.cv_y_pred_ = best_y_pred
        
        self.is_fitted = True

        X_offset += X_mean * X_scale
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - torch.matmul(X_offset, self.coef_.T)
        else:
            self.intercept_ = torch.zeros(1, device=self.coef_.device)

        return self
    
    def predict(self, X):
        X = self.parse_input_data(X)
        return X.matmul(self.coef_.T) + self.intercept_
    
    def score(self, X, y):
        X = self.parse_input_data(X)
        y = self.parse_input_data(y)
        return self.scorer(y, self.predict(X))