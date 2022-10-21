import os
import matplotlib.pyplot as plt
import bull_spread as bs
import numpy as np

if __name__ == '__main__':

    plot_fold = 'bull_spread_1month_2^18'

    maturities = [1., 1.0082137210209656, 1.016427442041931, 1.0246411630628967, 1.032854884083862, 1.0410686051048277, 1.049282326125793, 1.0574960471467587, 1.0657097681677241, 1.0739234891886897, 1.0821372102096551]
    BS_list = [0.05818268644827651, 0.05827241692404925, 0.05836098967884565, 0.058448425658289865, 0.05853474529425284, 0.05861996852076465, 0.05870411478933532, 0.0587872030837055, 0.05886925193405426, 0.0589502794306874, 0.059030303237227494]
    upper_list = [0.05818268644827651, 0.05862820129801663, 0.058921828547823074, 0.059188123470682405, 0.05943039081931198, 0.05963484543038024, 0.05986586614126655, 0.06000955861342226, 0.06022169057200912, 0.06044814020561235, 0.0605981063367854]
    lower_list = [0.05818268644827651, 0.05782488990960281, 0.05754561786627617, 0.057367093077278074, 0.057156683742470314, 0.05700077320433377, 0.05687350222230735, 0.056706520125267945, 0.05655756062910731, 0.05643600281648584, 0.056260512102328235]


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