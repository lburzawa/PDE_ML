import numpy as np


def set_ode_fun(parameters):

    def fun(t, y):
        BMP = y[0*parameters.n : 1*parameters.n]
        Chd = y[1*parameters.n : 2*parameters.n]
        Nog = y[2*parameters.n : 3*parameters.n]
        Szd = y[3*parameters.n : 4*parameters.n]
        BMPChd = y[4*parameters.n : 5*parameters.n]
        BMPNog = y[5*parameters.n : 6*parameters.n]

        qa = (t - 6480.0) / (2000.0 + abs(t - 6480.0))
        q = qa * float(qa > 0)
        Szd_factor1 = 1 + Szd / parameters.kit + (Chd + BMPChd) / parameters.kmt
        Szd_factor2 = 1 + Szd / parameters.kia + (Chd + BMPChd) / parameters.kma
        lambda_Tld_Chd_space = q * parameters.lambda_Tld_Chd * parameters.yspace * Chd / Szd_factor1
        lambda_Tld_BMPChd_space = q * parameters.lambda_Tld_BMPChd * parameters.yspace * BMPChd / Szd_factor1
        lambda_bmp1a_Chd_space = parameters.lambda_bmp1a_Chd * Chd / Szd_factor2
        lambda_bmp1a_BMPChd_space = parameters.lambda_bmp1a_BMPChd * BMPChd / Szd_factor2

        eta_BMP = parameters.j1 * parameters.yspace
        eta_Chd = np.zeros((parameters.n,), dtype=np.float64)
        eta_Nog = np.zeros((parameters.n,), dtype=np.float64)
        eta_Chd[parameters.n-parameters.ndor_Chd:] = parameters.j2
        eta_Nog[parameters.n-parameters.ndor_Nog:] = parameters.j3

        BMP_Chd_merge = parameters.k1 * BMP * Chd
        BMP_Nog_merge = parameters.k2 * BMP * Nog
        BMPChd_split = parameters.k_1 * BMPChd
        BMPNog_split = parameters.k_2 * BMPNog
        BMP_pow_nu = np.power(BMP, parameters.nu)
        kernel = np.array([1.0, -2.0, 1.0])
        diff = np.correlate(BMP, kernel, 'same')
        diff[0] += BMP[1]
        diff[-1] += BMP[-2]
        dBMPdt = parameters.D_BMP * diff - BMP_Chd_merge + BMPChd_split - BMP_Nog_merge + BMPNog_split - parameters.dec_BMP * BMP + eta_BMP \
                    + lambda_Tld_BMPChd_space + lambda_bmp1a_BMPChd_space
        diff = np.correlate(Chd, kernel, 'same')
        diff[0] += Chd[1]
        diff[-1] += Chd[-2]
        dChddt = parameters.D_Chd * diff - BMP_Chd_merge + BMPChd_split - parameters.dec_Chd * Chd + eta_Chd \
                    - lambda_Tld_Chd_space - lambda_bmp1a_Chd_space
        diff = np.correlate(Nog, kernel, 'same')
        diff[0] += Nog[1]
        diff[-1] += Nog[-2]
        dNogdt = parameters.D_Nog * diff - BMP_Nog_merge + BMPNog_split - parameters.dec_Nog * Nog + eta_Nog
        diff = np.correlate(Szd, kernel, 'same')
        diff[0] += Szd[1]
        diff[-1] += Szd[-2]
        dSzddt = parameters.D_Szd * diff - parameters.dec_Szd * Szd + (parameters.Vs * BMP_pow_nu) / (parameters.k + BMP_pow_nu)
        diff = np.correlate(BMPChd, kernel, 'same')
        diff[0] += BMPChd[1]
        diff[-1] += BMPChd[-2]
        dBMPChddt = parameters.D_BMPChd * diff + BMP_Chd_merge - BMPChd_split - parameters.dec_BMPChd * BMPChd \
                       - lambda_Tld_BMPChd_space - lambda_bmp1a_BMPChd_space
        diff = np.correlate(BMPNog, kernel, 'same')
        diff[0] += BMPNog[1]
        diff[-1] += BMPNog[-2]
        dBMPNogdt = parameters.D_BMPNog * diff + BMP_Nog_merge - BMPNog_split - parameters.dec_BMPNog * BMPNog

        dydt = np.concatenate((dBMPdt, dChddt, dNogdt, dSzddt, dBMPChddt, dBMPNogdt), 0)

        return dydt

    return fun
