import numpy.polynomial.polynomial as poly
import numpy as np 
import warnings

m_0 = 0.511*1e-3 # mass in [GeV]
d = 2.26 # [m] distance between Q525 and OTR2
l = 0.108 # effective length [m]

def get_gradient(b_field, l_eff=0.108):
    ''' Returns the quad field gradient [T/m] 
        l_eff: effective length [m] 
        b_field: integrated field [kG] '''
    return np.array(b_field)*0.1 /l_eff
    
def get_k1(g, p):
    '''Returns quad strength [1/m^2]
       g: quad field gradient [T/m]
       p: momentum [GeV] (or alternatively beta*E [GeV])'''
    return 0.2998 * g / p

def fit_sigma(sizes, k, d=d, l=l, show_plots=False):
    '''Fit sizes^2 = c0 + c1*k + c2*k^2
       returns: c0, c1, c2 '''
    coefs = poly.polyfit(k, sizes**2, 2)
   
    # matrix elements at quad 525
    c0, c1, c2 = coefs
    #return c0,c1,c2
    sig_11 = c2 / (d*l)**2
    sig_12 = (-c1 - 2*d*l*sig_11) / (2*d**2*l)
    sig_22 = (c0 - sig_11 - 2*d*sig_12) / d**2
    
    ### plotting 
#     import matplotlib.pyplot as plt
#     import datetime
#     fig = plt.figure(figsize=(7,5))
#     ffit = np.poly1d(coefs)
#     xfit = np.linspace(np.min(k),np.max(k),100)
#     plt.plot(k, sizes**2, marker="x")
#     plt.plot(xfit, poly.polyval(xfit, coefs))
#     plt.xlabel("k (1/m^2)")
#     plt.ylabel("sizes^2 (m^2)")
#     timestamp = (datetime.datetime.now()).strftime("%m-%d_%H-%M-%S")
#     plt.savefig(f"./plots/emittance_fit_{timestamp}.png")
#     if show_plots:
#         plt.show()
    ### end plotting
    
    return sig_11, sig_12, sig_22

def get_emit(sig11, sig12, sig22):
    '''Returns emittance (not normalized)'''
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emit  = np.sqrt(sig11*sig22 - sig12**2)
            return emit
    except RuntimeWarning:
        return 1000 # arbitrary high value, maybe the best way to handle this?

def get_normemit(energy, quad_vals, xrms, yrms, show_plots=False):
    '''Returns normalized emittance [m]
       given quad values and beamsizes'''
    gamma = energy/m_0
    beta = np.sqrt(1-1/gamma**2)

    b1_gradient = get_gradient(quad_vals)
    k = get_k1(b1_gradient, beta*energy)

    sig_11, sig_12, sig_22 = fit_sigma(np.array(xrms), k, show_plots=show_plots)
    emitx = get_emit(sig_11, sig_12, sig_22)

    sig_11, sig_12, sig_22 = fit_sigma(np.array(yrms), -k, show_plots=show_plots)
    emity = get_emit(sig_11, sig_12, sig_22)

    if emitx == 1000 or emity == 1000:
        return 1000

    return np.sqrt(emitx*emity)*gamma*beta