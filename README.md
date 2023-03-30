Classification and Authentication of Materials using Prompt Gamma Ray Activation Analysis
---
This repository accompanies the manuscript: [Classification and Authentication of Materials using Prompt Gamma Ray Activation Analysis]().

Prompt gamma ray activation analysis (PGAA) is a nuclear measurement technique that enables quantification of the number and type of different isotopes present in a sample.  Its non-destructive nature makes PGAA an attractive method for the potentially rapid identification of many real-world materials.  A typical gamma ray energy spectrum contains isotopic information on nearly all elements in the sample, providing an opportunity for robust material classification. Here, we use PGAA spectra to train different types of models to elucidate how discriminating these spectra can be for various classes of materials, and how well different types of models perform.  We trained discriminative models appropriate for closed set scenarios, where all possible classes a candidate material may belong to can be sampled.  We also trained various class models, such as DD-SIMCA, to better address open set conditions, where it is not possible to sample all possibilities.  With appropriate pre-processing and data treatments, all such models could be made to perform nearly perfectly on our dataset.  This suggests PGAA spectra may serve as a powerful nuclear fingerprint for high-throughput material classification.

Highlights:

* PGAA spectra were used to train predictive machine learning models to accurately identify a broad range of materials.
* Random forest models performed the best and are applicable when all relevant materials have been enumerated.
* Class models can achieve comparable performance, which are more appropriate under real-world, "open set" conditions.


Installation
---
Notebooks are included to reproduce calculations and figures in the manuscript.  They have the following dependencies:

Set up the conda environment for this project.
```code
$ conda env create -f conda-env.yml
$ conda activate pgaa-authentication
$ python -m ipykernel install --user --name=pgaa-authentication
```

Install [pychemauth](https://github.com/mahynski/pychemauth) which is a dependency.
```code
$ conda activate pgaa-authentication
$ git clone https://github.com/mahynski/pychemauth.git --branch v0.0.0b3 --depth 1
$ cd pychemauth
$ pip install .
```

Contributors
---
Nathan A. Mahynski

Jacob I. Monroe

David A. Sheen

Rick L. Paul

H. Heather Chen-Mayer

Vincent K. Shen
