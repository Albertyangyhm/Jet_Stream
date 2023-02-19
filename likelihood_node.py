import numpy as np
import torch
from scipy.special import logsumexp
import copy


def enrich_jet_logLH(rootNode, delta_min=None, dij=False, alpha = None):
    """
    Attach splitting log likelihood to each edge, by calling recursive
    _get_jet_likelihood.
    """
    def _get_jet_logLH(
        currNode
    ):
        """
        Recursively enrich every edge from root_id downward with their log likelihood.
        log likelihood of a leaf is 0. Assumes a valid currNode.
        """

        nonlocal delta_min, dij


        if not currNode.left or not currNode.right:
            currNode.logLH = 0
            return

        pL = currNode.left.vec4
        tL = currNode.left.delta
        pR = currNode.right.vec4
        tR = currNode.right.delta


        Lambda = currNode.decay_rate

        # W and QCD decay root special case not handeled yet
        # if root_id == currNode["root_id"]:
        #     Lambda = currNode["LambdaRoot"]


        llh = split_logLH_with_stop_nonstop_prob(pL,  pR,  delta_min, Lambda)
        currNode.logLH = llh

        if dij:
            dijs= [float(llh)]

            for alpha in [-1,0,1]:

                tempCos = np.dot(pL[1::], pR[1::]) / (np.linalg.norm(pL[1::]) * np.linalg.norm(pR[1::]))
                if abs(tempCos) > 1: tempCos = np.sign(tempCos)

                dijVal = np.sort((np.array([np.linalg.norm(pL[1:3]),np.linalg.norm(pR[1:3])])) ** (2 * alpha))[0]  * \
                        (
                            np.arccos(tempCos)
                        ) ** 2

                dijs.append(dijVal)

            currNode.dijList = dijs


        _get_jet_logLH(
            currNode.left
        )
        _get_jet_logLH(
            currNode.right
        )
    
    _get_jet_logLH(
        rootNode
    )

    return rootNode




def split_logLH_with_stop_nonstop_prob(pL, pR, t_cut, lam):
    """
    Take two nodes and return the splitting log likelihood
    """
    tL = pL[0] ** 2 - np.linalg.norm(pL[1::]) ** 2
    tR = pR[0] ** 2 - np.linalg.norm(pR[1::]) ** 2


    pP = pR + pL ## eq (5)


    """Parent invariant mass squared"""
    tp = pP[0] ** 2 - np.linalg.norm(pP[1::]) ** 2

    if tp<=0 or tL<0 or tR<0:
        return - np.inf

    # print("tP = ", tp, " tL = ", tL, " | tR= ", tR)
    # print("lam= ",lam, " | pP = ", pP, " pL = ", pL, " | pR= ", pR)

    """ We add a normalization factor -np.log(1 - np.exp(- lam)) because we need the mass squared to be strictly decreasing. This way the likelihood integrates to 1 for 0<t<t_p. All leaves should have t=0, this is a convention we are taking (instead of keeping their value for t given that it is below the threshold t_cut)"""
    def get_logp(tP_local, t, t_cut, lam):


        if t > t_cut:
            """ Probability of the shower to stop F_s"""
            # F_s = 1 / (1 - np.exp(- lam)) * (1 - np.exp(-lam * t_cut / tP_local))
            # if F_s>=1:
            #     print("Fs = ", F_s, "| tP_local = ", tP_local, "| t_cut = ", t_cut, "| t = ",t)

            # print("Inner - t = ",t," | tL =",tL, " | tR = ",tR," pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ",-np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local)
            # return -np.log(1 - np.exp(- lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local + np.log(1-F_s)
            return -np.log(1 - np.exp(- (1. - 1e-3)*lam)) + np.log(lam) - np.log(tP_local) - lam * t / tP_local

        else: # For leaves we have t<t_cut
            t_upper = min(tP_local,t_cut) #There are cases where tp2 < t_cut
            log_F_s = -np.log(1 - np.exp(- (1. - 1e-3)*lam)) + np.log(1 - np.exp(-lam * t_upper / tP_local))
            # print("Outer - t = ",t," | tL =",tL, " | tR = ",tR," pL = ", pL, " | pR= ", pR, " | pP = ", pP, "logLH = ", log_F_s)
            return log_F_s


    if tp <= t_cut:
        "If the pairing is not allowed"
        logLH = - np.inf

    elif tL >=(1 - 1e-3)* tp or tR >=(1 - 1e-3)* tp:
        # print("The pairing is not allowed because tL or tR are greater than tP")
        logLH = - np.inf

    elif np.sqrt(tL) + np.sqrt(tR) > np.sqrt(tp):
        print("Breaking invariant mass inequality condition")
        logLH = - np.inf


    else:
        """We sample a unit vector uniformly over the 2-sphere, so the angular likelihood is 1/(4*pi)"""

        tpLR = (np.sqrt(tp) - np.sqrt(tL)) ** 2
        tpRL = (np.sqrt(tp) - np.sqrt(tR)) ** 2

        logpLR = np.log(1/2)+ get_logp(tp, tL, t_cut, lam) + get_logp(tpLR, tR, t_cut, lam) #First sample tL
        logpRL = np.log(1/2)+ get_logp(tp, tR, t_cut, lam) + get_logp(tpRL, tL, t_cut, lam) #First sample tR

        logp_split = logsumexp(np.asarray([logpLR, logpRL]))

        logLH = (logp_split + np.log(1 / (4 * np.pi)) ) ## eq (8)

    return logLH