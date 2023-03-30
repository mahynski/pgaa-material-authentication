# LDA Model

import imblearn
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pipe_preprocessing import preprocessing_factory
from pipe_preprocessing import get_nested_scores

def run(X_train, y_train, k_inner, k_outer):
    performance_summary = {}

    iter_ = 0
    n_classes = len(np.unique(y_train))
    for threshold in [True]:
        for scaling in [None, 'standard', 'robust']:
            for dim_red in [None, 'pca', 'pacmap']:
                name = 'lda'+('_thresh' if threshold else '')+('_'+scaling if scaling is not None else '')+('_'+dim_red if dim_red is not None else '')
                try:
                    steps, param_grid = preprocessing_factory(balance=True,
                                                              savgol=False,
                                                              threshold=threshold,
                                                              scaling=scaling,
                                                              dim_red=dim_red)

                    steps += [('model', LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None))]
                    param_grid.update({
                        'model__n_components':np.linspace(1, (n_classes-1)+1, 5, dtype=int),
                        'model__tol':[1.0e-8, 1.0e-4],
                    })

                    performance_summary[name] = get_nested_scores(
                        imblearn.pipeline.Pipeline(steps=steps), 
                        [param_grid], 
                        X_train, 
                        y_train, 
                        k_inner=k_inner, 
                        k_outer=k_outer
                    )
                except:
                    print('***FAILED: {}***'.format(name)) 

                iter_ += 1
                print('Finished iter: {}'.format(iter_))
                
    return performance_summary

if __name__ == '__main__':
    X = pickle.load(open('../../raw/X_use.pkl', 'rb'))
    y = pickle.load(open('../../raw/y_use.pkl', 'rb'))
    performance_summary = run(X, y, k_inner=2, k_outer=5)
    pickle.dump(performance_summary, open('lda_summary.pkl', 'wb'), protocol=4)
    
