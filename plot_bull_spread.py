import os
import matplotlib.pyplot as plt
import bull_spread as bs
import numpy as np

if __name__ == '__main__':

    plot_fold = 'bull_spread_1month_2^19'

    maturities = [1., 1.0082137210209656, 1.016427442041931, 1.0246411630628967, 1.032854884083862, 1.0410686051048277, 1.049282326125793, 1.0574960471467587, 1.0657097681677241, 1.0739234891886897, 1.0821372102096551]
    BS_list = [0.05818268644827651, 0.05827241692404925, 0.05836098967884565, 0.058448425658289865, 0.05853474529425284, 0.05861996852076465, 0.05870411478933532, 0.0587872030837055, 0.05886925193405426, 0.0589502794306874, 0.059030303237227494]
    upper_list = [0.05818268644827651, 0.05896676764531234, 0.059466812917708854, 0.05991837403694702, 0.06031778742215704, 0.060697498311321235, 0.06099513213238053, 0.061356750743898514, 0.06169973763880863, 0.06200163752356896, 0.06229827328586948]
    lower_list = [0.05818268644827651, 0.05752757182616754, 0.05709626128911022, 0.056764452765395025, 0.056419310806496356, 0.05611901436333489, 0.055883783159853255, 0.055602890200265785, 0.0553375283540108, 0.055148843603758636, 0.05486583578205415]


    # plots

    plt.plot(maturities, upper_list, label='Upper level')
    plt.plot(maturities, lower_list, label='Lower level')
    plt.plot(maturities, BS_list, label='B&S')
    plt.xlabel('Maturity')
    plt.ylabel('Fair value')
    plt.xlim([maturities[0] - 1./365.2425, maturities[-1] + 1./365.2425])
    plt.ylim([0.048, 0.0670])
    plt.legend()
    plt.savefig(os.path.join('plots', plot_fold, 'option_levels_v2.png'), bbox_inches='tight')
    plt.clf()

    plt.plot(maturities, upper_list, label='Upper level')
    plt.plot(maturities, lower_list, label='Lower level')
    plt.plot(maturities, BS_list, label='B&S')
    plt.xlabel('Maturity')
    plt.ylabel('Fair value')
    plt.xlim([0., maturities[-1] + 1. / 365.2425])
    plt.ylim([0.04, 0.0675])
    plt.legend()
    plt.savefig(os.path.join('plots', plot_fold, 'option_levels_v3.png'), bbox_inches='tight')
    plt.clf()

    maturities_full = np.arange(1.,  396) / 365
    bs_full = bs.bull_call_spread(1, 1, 1.2, maturities_full, 0, 0, 0.2)
    plt.plot(maturities, upper_list, label='Upper level')
    plt.plot(maturities, lower_list, label='Lower level')
    plt.plot(maturities_full, bs_full, label='B&S')
    plt.xlabel('Maturity')
    plt.ylabel('Fair value')
    plt.xlim([0., maturities[-1] + 1. / 365.2425])
    plt.ylim([0, 0.0675])
    plt.legend()
    plt.savefig(os.path.join('plots', plot_fold, 'option_levels_v4.png'), bbox_inches='tight')
    plt.clf()