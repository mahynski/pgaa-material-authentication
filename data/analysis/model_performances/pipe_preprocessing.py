"""
Basic preprocessing part of pipelines and other tools.
"""

import pychemauth
import pacmap
import sklearn

from pychemauth.preprocessing.imbalanced import ScaledSMOTEENN
from pychemauth.preprocessing.filter import SavGol
from pychemauth.preprocessing.scaling import CorrectedScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from pychemauth.analysis.cross_validation import NestedCV

def preprocessing_factory(balance=False, savgol=False, threshold=False, scaling='standard', dim_red='pca'):
    """A simple factory to produce different pre-processing steps."""
    steps = []
    param_grid = {}
    
    if threshold:
        steps += [
            ('variance_threshold', VarianceThreshold(
                threshold=1.0)
            )
        ]
        param_grid.update({
            'variance_threshold__threshold':[1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4],
        })
    
    if balance: # Balancer does scaling (with std) so remove "bad" columns first
        steps += [
            ('class_balancer', ScaledSMOTEENN(
                random_state=42,
                k_enn=3,
                k_smote=3,
                kind_sel_enn='all')
            )
        ]
        param_grid.update({
            'class_balancer__k_enn':[3, 5], 
            'class_balancer__k_smote':[3, 5], 
            'class_balancer__kind_sel_enn':['all'], 
        })
        
    if savgol:
        steps += [
            ('filter', SavGol(
                window_length=5,
                polyorder=1, # You can vary this, but using non-linear curves can confuse SHAP
                deriv=0)
            )
        ]
        param_grid.update({
            'filter__window_length':[5, 11, 21],
            'filter__polyorder':[1, 2, 3],
            'filter__deriv':[0, 1],
        })
        
    if scaling is None:
        pass
    elif scaling == 'standard':
        steps += [
            ('scaling', CorrectedScaler(
                with_mean=True, # Always center and scale if bothering to use
                with_std=True,
                pareto=False,
                biased=False)
            )
        ]
        param_grid.update({
            'scaling__pareto':[True, False]
        })
    elif scaling == 'robust':
        steps += [
            ('scaling', RobustScaler(
                with_median=True, # Always center and scale if bothering to use
                with_iqr=True,
                pareto=False)
            )
        ]
        param_grid.update({
            'scaling__pareto':[True, False]
        })
    else:
        raise ValueError('Unrecognized scaling method: '+str(scaling))
        
    if dim_red is None:
        pass
    elif dim_red == 'pca':
        # sklearn's PCA automatically centers data on its own
        steps += [
            ('dim_red', PCA(
                n_components=1,
                svd_solver='arpack',
                tol=1.0e-6)
            )
        ]
        param_grid.update({
            'dim_red__n_components':[2, 3, 4, 5, 10, 20] # Keep relatively small, if using PCA
        })
    elif dim_red == 'pacmap':
        steps += [
            ('dim_red', pacmap.PaCMAP(
                n_components=3, 
                n_neighbors=5,
                MN_ratio=0.5, 
                FP_ratio=2.0,
                distance='euclidean',
                apply_pca=False, 
                lr=1.0,
                random_state=42)
            )
        ]
        param_grid.update({
            'dim_red__n_components':[2, 3], # Keep in the range of easy visualization, if using
            'dim_red__n_neighbors':[3, 5, 10],
            'dim_red__MN_ratio':[0.2, 0.5, 0.7],
            'dim_red__FP_ratio':[2.0, 4.0],
            'dim_red__distance':['euclidean', 'manhattan', 'angular'],
            'dim_red__lr':[0.01, 0.1, 1.0],
        })
    else:
        raise ValueError('Unrecognized dim_red method: '+str(dim_red))
    
    return steps, param_grid


def get_nested_scores(pipeline, param_grid, X, y, k_inner, k_outer):
    """Nested CV for statistical comparisons."""
    NCV = NestedCV(k_inner=k_inner, k_outer=k_outer)
    return NCV.grid_search(pipeline, param_grid, X, y, classification=True)

def final_fit(X, y, pipeline, param_grid, k):
    """CV on entire dataset, retrain on full set at the end."""
    
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        n_jobs=-1,
        cv=sklearn.model_selection.StratifiedKFold(
            n_splits=k, 
            shuffle=True, 
            random_state=0,
        ),
        error_score=0,
        refit=True # Re-fit the model on the entire dataset
    )
    
    gs.fit(X, y) 
    
    return gs

if __name__ == '__main__':
    print(__file__)
    
