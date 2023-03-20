# import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()

def llhTF(pL, pR, t_cut, lam):
    """
    Take two nodes and return the splitting log likelihood
    """
    tL = pL[0] ** 2 - tf.norm(pL[1::]) ** 2
    tR = pR[0] ** 2 - tf.norm(pR[1::]) ** 2


    pP = pR + pL ## eq (5)


    # Parent invariant mass squared
    tp = pP[0] ** 2 - tf.norm(pP[1::]) ** 2

    if tp<=0 or tL<0 or tR<0:
        return - tf.math.inf

    # We add a normalization factor -tf.math.log(1 - tf.math.exp(- lam))
    # because we need the mass squared to be strictly decreasing.
    # This way the likelihood integrates to 1 for 0<t<t_p.
    # All leaves should have t=0, this is a convention we are
    # taking (instead of keeping their value for t given that
    # it is belofw the threshold t_cut)
    def get_logp(tP_local, t, t_cut, lam):
        if t > t_cut:
            # Probability of the shower to stop F_s
            return -tf.math.log(1 - tf.math.exp(- (1. - 1e-3)*lam))\
                 + tf.math.log(lam) - tf.math.log(tP_local) - lam * t\
                     / tP_local

        else: # For leaves we have t<t_cut
            t_upper = tf.minimum(tP_local,t_cut) #There are cases where tp2 < t_cut
            log_F_s = -tf.math.log(1 - \
                tf.math.exp(- (1. - 1e-3)*lam)) +\
                     tf.math.log(1 - tf.math.exp(-lam\
                         * t_upper / tP_local))
            return log_F_s


    if tp <= t_cut:
        #If the pairing is not allowed
        logLH = - tf.math.inf

    elif tL >=(1 - 1e-3)* tp or tR >=(1 - 1e-3)* tp:
        # print("The pairing is not allowed because tL or tR are greater than tP")
        logLH = - tf.math.inf

    elif tf.sqrt(tL) + tf.sqrt(tR) > tf.sqrt(tp):
        print("Breaking invariant mass inequality condition")
        logLH = - tf.math.inf


    else:
        # We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)

        tpLR = (tf.sqrt(tp) - tf.sqrt(tL)) ** 2
        tpRL = (tf.sqrt(tp) - tf.sqrt(tR)) ** 2

        logpLR = tf.math.log(1/2)+ get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL
        logpRL = tf.math.log(1/2)+ get_logp(tp, tR, t_cut, lam) + get_logp(tpRL, tL, t_cut, lam) #First sample tR

        logp_split = tf.reduce_logsumexp(tf.stack([logpLR, logpRL]))
        logLH = logp_split + tf.math.log(1 / (4 * tf.constant(np.pi)))
        
    return logLH, tp