#!/bin/bash
source /home/$USER/anaconda3/bin/activate;
conda activate pgaa-authentication;

for i in run_dtree run_lda run_qda run_rf run_logreg run_simca_compliant run_simca_rigorous run_soft_plsda run_hard_plsda; do
	python $i.py 2> $i.err > $i.log;
done;

