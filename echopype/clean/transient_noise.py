"""
Algorithms for masking transient noise.

    Copyright (c) 2020 Echopy

    This code is licensed under the MIT License.
    See https://opensource.org/license/mit/ for details.
    Original code sourced from:
    https://github.com/open-ocean-sounding/echopy/blob/master/echopy/processing/mask_transient.py

"""
__authors__ = [
    "Alejandro Ariza",  # wrote ryan(), fielding()
    "Mihai Boldeanu",  # adapted the mask transient noise algorithms to echopype
]


import numpy as np

from ..utils.mask_transformation import lin as _lin, log as _log


def _ryan(Sv, r, m=50, n=20, thr=20, excludeabove=250, operation="percentile15"):
    """
    Mask transient noise as in:

        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    This mask is based on the assumption that Sv values which exceed the median
    value in a surrounding region of m metres by n pings must be due to
    transient noise. Sv values are removed if exceed a threshold. Masking is
    excluded above 250 m by default to avoid the removal of aggregated biota.

    Args:
        Sv (float): 2D numpy array with Sv data to be masked (dB)
        r (float): 1D numpy array with range data (m)
        m (int): height of surrounding region (m)
        n (int): width of surrounding region (pings)
        threshold (int): user-defined threshold for comparisons (dB)
        excludeabove (int): range above which masking is excluded (m)
        operation (str): type of average operation:
            'mean'
            'percentileXX'
            'median'
            'mode'#not in numpy

    Returns:
        bool: 2D numpy array mask (transient noise = True)
    """
    # offsets for i and j indexes
    ioff = n
    joff = np.argmin(abs(r - m))

    # preclude processing above a user-defined range
    r0 = np.argmin(abs(r - excludeabove))

    # mask if Sv sample greater than averaged block
    # TODO: find out a faster method. The iteration below is too slow.
    mask = np.ones(Sv.shape, dtype=bool)
    mask[:, 0:r0] = False

    for i in range(len(Sv)):
        for j in range(r0, len(Sv[0])):
            # proceed only if enough room for setting the block
            if (i - ioff >= 0) & (i + ioff < len(Sv)) & (j - joff >= 0) & (j + joff < len(Sv[0])):
                sample = Sv[i, j]
                if operation == "mean":
                    block = _log(np.nanmean(_lin(Sv[i - ioff : i + ioff, j - joff : j + joff])))
                elif operation == "median":
                    block = _log(np.nanmedian(_lin(Sv[i - ioff : i + ioff, j - joff : j + joff])))
                else:
                    block = _log(
                        np.nanpercentile(
                            _lin(Sv[i - ioff : i + ioff, j - joff : j + joff]), int(operation[-2:])
                        )
                    )
                mask[i, j] = sample - block > thr

    return mask


def _fielding(Sv, r, r0=200, r1=1000, n=20, thr=[2, 0], roff=250, jumps=5, maxts=-35, start=0):
    """
    Mask transient noise with method proposed by Fielding et al (unpub.).

    A comparison is made ping by ping with respect to a block in a reference
    layer set at far range, where transient noise mostly occurs. If the ping
    median is greater than the block median by a user-defined threshold, the
    ping will be masked all the way up, until transient noise disappears, or
    until it gets the minimum range allowed by the user.

       transient                 transient             ping
         noise                     noise             evaluated
           |                         |                  |
    ______ | _______________________ | ____________.....V.....____________
          |||  far range interval   |||            .  block  .            |
    _____|||||_____________________|||||___________...........____________|

    When transient noise is detected, comparisons start to be made in the same
    ping but moving vertically every x meters (jumps). Pings with transient
    noise will be masked up to where the ping is similar to the block according
    with a secondary threshold or until it gets the exclusion range depth.

    Args:
        Sv    (float): 2D numpy array with Sv data to be masked (dB).
        r     (float): 1D numpy array with range data (m).
        r0    (int  ): range below which transient noise is evaluated (m).
        r1    (int  ): range above which transient noise is evaluated (m).
        n     (int  ): n of preceding & subsequent pings defining the block.
        thr   (int  ): user-defined threshold for side-comparisons (dB).
        roff  (int  ): range above which masking is excluded (m).
        maxts (int  ): max transient noise permitted, prevents to interpret
                       seabed as transient noise (dB).
        jumps (int  ): height of vertical steps (m).
        start (int  ): ping index to start processing.

    Returns:
        list: 2D boolean array with TN mask and 2D boolean array with mask
              indicating where TN detection was unfeasible.
    """

    # raise errors if wrong arguments
    if r0 > r1:
        raise Exception("Minimum range has to be shorter than maximum range")

    # return empty mask if searching range is outside the echosounder range
    if (r0 > r[-1]) or (r1 < r[0]):
        mask = np.zeros_like(Sv, dtype=bool)
        mask_ = np.zeros_like(Sv, dtype=bool)
        return mask, mask_

    # get upper and lower range indexes
    up = np.argmin(abs(r - r0))
    lw = np.argmin(abs(r - r1))

    # get minimum range index admitted for processing
    rmin = np.argmin(abs(r - roff))
    # get scaling factor index
    sf = np.argmin(abs(r - jumps))
    # start masking process
    mask_ = np.zeros(Sv.shape, dtype=bool)
    mask = np.zeros(Sv.shape, dtype=bool)
    for j in range(start, len(Sv)):
        # mask where TN evaluation is unfeasible (e.g. edge issues, all-NANs)
        if (j - n < 0) | (j + n > len(Sv) - 1) | np.all(np.isnan(Sv[j, up:lw])):
            mask_[j, :] = True
        # evaluate ping and block averages otherwise

        else:
            pingmedian = _log(np.nanmedian(_lin(Sv[j, up:lw])))
            pingp75 = _log(np.nanpercentile(_lin(Sv[j, up:lw]), 75))
            blockmedian = _log(np.nanmedian(_lin(Sv[j - n : j + n, up:lw])))

            # if ping median below 'maxts' permitted, and above enough from the
            # block median, mask all the way up until noise disappears
            if (pingp75 < maxts) & ((pingmedian - blockmedian) > thr[0]):
                r0, r1 = lw - sf, lw
                while r0 > rmin:
                    pingmedian = _log(np.nanmedian(_lin(Sv[j, r0:r1])))
                    blockmedian = _log(np.nanmedian(_lin(Sv[j - n : j + n, r0:r1])))
                    r0, r1 = r0 - sf, r1 - sf
                    if (pingmedian - blockmedian) < thr[1]:
                        break
                mask[j, r0:] = True

    final_mask = mask[:, start:] | mask_[:, start:]
    return final_mask
