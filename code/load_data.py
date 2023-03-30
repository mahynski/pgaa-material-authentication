"""
Utility functions for reading in PGAA spectra.

author: nam
"""
import numpy as np
import tqdm
import scipy.interpolate

def align_spectra(spectra, energies, coarsen=1, drop=0):
    """
    Aligns spectra so that energy bins line up. In other words, need bin i to
    correspond to the same energy for all spectra.

    Parameters
    ----------
    spectra : ndarray-like
        raw spectra, or counts at each energy bin
    energies : ndarray-like
        energy bins for raw spectra
    coarsen : int (4)
        number of bins to combine together when aligning to identify new energies
    drop : int (40)
        number of bins to drop from beginning (low energies) after alignment
        (peaks below ~100 keV are not reliable, so set to exclude)

    Returns
    -------
    aligned_spectra, aligned_center : ndarray, ndarray
        Spectra (counts) at the aligned energies provided in aligned_center

    """

    lowest_energy = np.min(energies)
    highest_energy = np.max(energies)
    nbins = int(spectra.shape[1]) # Raw data all have same number of bins, just different centers
    dbin = (highest_energy - lowest_energy) / nbins

    # [LB c0 | c1 | c2 | ... | cn UB]
    bin_centers = np.arange(nbins)*dbin + dbin/2.0 + lowest_energy
    aligned_centers = np.array(
        [np.mean(bin_centers[coarsen*start:coarsen*(start+1)])
         for start in range(len(bin_centers)//coarsen)],
        dtype=np.float32
    )

    aligned_spectra = []

    def get_bin(e):
        return np.array(np.floor((e-lowest_energy)/dbin), dtype=int)

    # Align and coarsen the spectra
    for s,e in tqdm.tqdm(zip(spectra, energies)):
        # Linear interpolation for stability; sets to 0 outside of range
        f = scipy.interpolate.interp1d(x=e, y=s, kind='linear', fill_value=(s[0], s[-1]), bounds_error=False)
        interp_spec = f(bin_centers)

        coarsened = np.array(
            [np.sum(interp_spec[coarsen*start:coarsen*(start+1)]) 
             for start in range(len(interp_spec)//coarsen)], 
            dtype=np.float64
        )
        aligned_spectra.append(coarsened)
    aligned_spectra = np.array(aligned_spectra)

    # Remove low-energy bins
    aligned_spectra = aligned_spectra[:, drop:]
    aligned_centers = aligned_centers[drop:]

    return aligned_spectra, aligned_centers

def read_spe(filename, normalize=None, convert=True, annihilation=False, coarsen=1):
    """
    Read raw SPE files.

    Parameters
    ----------
    filename : str
    	Name of SPE file to read.
    normalize : str
        Divides the spectra by the total counts.
    convert : bool
        Convert bin numbers to energy.
    annihilation : bool
        Remove peak at 511 keV due to positron annihilation. This will only
        work if convert is set to True.
    coarsen : int
        Number of neighboring bins to combine (sum) into a single bin. Default of
        1 will not change the input.

    Returns
    -------
    spectra, bins : ndarray, ndarray
        Energy spectrum and (centered) bins (in energy units if converted) they correspond to.
    """
    with open(filename,'r') as f:
        contents = f.read().split('\n')

    first = np.where(['$DATA:' == x for x in contents])[0][0]+2
    last = np.where(['$ENER_FIT:' == x for x in contents])[0][0]
    times = contents[np.where(['$MEAS_TIM:' == x for x in contents])[0][0]+1].split(' ')
    lifetime = float(times[0])
    runtime = float(times[1])
    deadtime = runtime - lifetime
    intercept = float(contents[last+1].split(' ')[0])
    slope = float(contents[last+1].split(' ')[1])

    spectra = np.array(contents[first:last], dtype=np.float64)

    # 1. Convert bins to energy values
    bins = np.arange(len(spectra), dtype=np.float64)
    if convert:
        bins = bins*slope + intercept

    # 2. Normalization
    if normalize == 'empirical':
        spectra = spectra / np.sum(spectra)
    elif normalize is None:
        pass
    else:
        raise ValueError("Input %s to keyword argument normalize not understood. Must be either 'physics' or 'empirical'"%normalize)

    # 3. Remove annihilation peak before coarsening
    positron_peak = 511.0
    if annihilation and convert:
        spectra[np.argmin(np.abs(bins-positron_peak))] = 0.0

    # 4. Coarsen bins by summing the measured counts
    spectra = np.array([np.sum(spectra[coarsen*start:coarsen*(start+1)]) 
                        for start in range(len(spectra)//coarsen)], dtype=np.float64)

    if convert:
        # Average the energy units
        bins = np.array([np.mean(bins[coarsen*start:coarsen*(start+1)]) 
                         for start in range(len(bins)//coarsen)], dtype=np.float64)
    else:
        # Renumber as integers
        bins = np.arange(len(spectra), dtype=np.float64)

    return spectra, bins


