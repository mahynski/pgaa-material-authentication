# SIMCA Model using a rigorous approach

import pychemauth
import imblearn
import pickle
import numpy as np
from pychemauth.classifier.simca import SIMCA_Classifier
from pipe_preprocessing import preprocessing_factory
from pipe_preprocessing import get_nested_scores

prefix = 'simca_rigorous'

def run(X_train, y_train, k_inner, k_outer):
    performance_summary = {}

    for i, target_class in enumerate(sorted(np.unique(y_train))):
        iter_ = 0
        for threshold in [True]:
            for dim_red in [None]: # SIMCA uses PCA internally so no sense in doing it twice
                if dim_red == 'pacmap': # Pacmap might benefit from other scalings beforehand, though
                    scalings = [None, 'standard', 'robust']
                else:
                    scalings = [None]
                for scaling in scalings:
                    name = prefix+'_class_{}'.format(i)+('_thresh' if threshold else '')+('_'+scaling if scaling is not None else '')+('_'+dim_red if dim_red is not None else '')
                    try:
                        steps, param_grid = preprocessing_factory(balance=True,
                                                                  savgol=False,
                                                                  threshold=threshold,
                                                                  scaling=scaling,
                                                                  dim_red=dim_red)

                        steps += [('model', SIMCA_Classifier(
                            target_class=target_class, 
                            style="dd-simca", 
                            use="rigorous", 
                            alpha=0.05, # Rigorous method fixes alpha and adjust other parameters to match this
                            robust=None,
                            sft=False))]
                        param_grid.update({
                            'model__scale_x':[True, False],
                            'model__n_components':np.arange(1, 10+1),
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
    pickle.dump(performance_summary, open(prefix+'_summary.pkl', 'wb'), protocol=4)
    
