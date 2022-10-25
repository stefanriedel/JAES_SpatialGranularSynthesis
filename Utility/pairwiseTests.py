import numpy as np
import scipy.stats as stats

def posthoc_wilcoxon(data, pairs_to_be_tested, p_adjust='BH'):
    pvals = np.zeros(len(pairs_to_be_tested))
    idx = 0
    for pair in pairs_to_be_tested:
        w, pvals[idx] = stats.wilcoxon(data[pair[0],:], data[pair[1],:])
        idx += 1

    if p_adjust == 'BH':
        sort_indices = np.argsort(pvals)
        corr_factor = pvals.size
        for i in range(pvals.size):
            pvals[sort_indices[i]] *= corr_factor

            if (pvals[sort_indices[i]] < pvals[sort_indices[i-1]]) and (i > 0):
                # EXIT: If a p-value is smaller than the previous after the correction,
                # meaning a change in the order due to correction, clip it to the corrected previous value.
                # This avoids a p-value to be significant when the previous/smaller value 
                # was insignificant after correction. This corresponds to an EXIT strategy.
                pvals[sort_indices[i]] = pvals[sort_indices[i-1]]

            # Clip p-values to 1.0
            if pvals[sort_indices[i]] > 1.0:
                pvals[sort_indices[i]] = 1.0

            # Update correction factor
            corr_factor -= 1

    return pvals

