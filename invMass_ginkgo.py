#!/usr/bin/env python

import numpy as np
import time
import sys
import os
import copy
import pickle
import torch
from torch import nn
import pyro
from ginkgo.pyro_simulator import PyroSimulator
from ginkgo import likelihood_invM as likelihood
from ginkgo import auxFunctions
from node import *


class Simulator(PyroSimulator):
    def __init__(self, jet_p=None, pt_cut=1.0, M_hard=None, Delta_0=None, num_samples=1, minLeaves =2 , maxLeaves=np.inf, maxNTry=20000 ):
        super(Simulator, self).__init__()
        self.pt_cut = pt_cut
        self.M_hard = M_hard
        self.Delta_0 = Delta_0
        self.num_samples = num_samples
        self.minLeaves = minLeaves
        self.maxLeaves = maxLeaves
        self.maxNTry = maxNTry

        self.jet_p = jet_p # 4d vector for root node

    # forward function gets called automatically when instance of simulator is created
    def forward(self, inputs):

        root_rate = inputs[0]
        decay_rate = inputs[1]

        # Define pyro distributions as global variables

        # Sample a unit vector uniformly over the 2-sphere
        globals()["multiNormal_dist"] = pyro.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))

        # Add a Bernoulli dist to randomly shuffle the L/R siblings
        globals()["Bernoulli_dist"] = pyro.distributions.Bernoulli(probs=0.5)

        globals()["root_dist"] = pyro.distributions.Exponential(root_rate)
        globals()["decay_dist"] = pyro.distributions.Exponential(decay_rate)

        jet_list = []

        i = 0
        while len(jet_list) < self.num_samples and i < self.maxNTry:

            # generate a tree
            tree, content, deltas, draws, leaves = _traverse(
                self.jet_p,
                delta_P=self.Delta_0,
                cut_off=self.pt_cut,
                rate=decay_rate,
            )
            jet = dict()
            jet["root_id"] = 0
            # -1 means that it is an unknown dimension. So however long the tree is
            jet["tree"] = np.asarray(tree).reshape(-1, 2)  # Labels for the nodes in the tree
            jet["content"] = np.array([np.asarray(c) for c in content])
            jet["LambdaRoot"] = root_rate
            jet["Lambda"] = decay_rate
            jet["Delta_0"] = self.Delta_0
            jet["pt_cut"] = self.pt_cut
            jet["algorithm"] = "truth"
            jet["deltas"] = np.asarray(deltas)
            jet["draws"] = np.asarray(draws)
            jet["leaves"] = np.array([np.asarray(c) for c in leaves])

            if self.minLeaves <= len(jet["leaves"]) < self.maxLeaves:
                # setting the mass of the root of the jet
                if self.M_hard:
                    jet["M_Hard"] = float(self.M_hard)

                # still calling online lib
                likelihood.enrich_jet_logLH(jet, dij=False)
                # still calling online lib
                ConstPhi, PhiDelta, PhiDeltaListRel = auxFunctions.traversePhi(jet, jet["root_id"], [], [],[])

                jet["ConstPhi"] = ConstPhi
                jet["PhiDelta"] = PhiDelta
                jet["PhiDeltaRel"] = PhiDeltaListRel



                jet_list.append(jet)

                if len(jet_list) % 1000 == 0:
                    print("Generated ", len(jet_list), "jets with ", self.minLeaves, "<=number of leaves<", self.maxLeaves)


            i += 1
            if i % 1000==0:
                print("Generated ", i, " jets")


        return jet_list

    # export
    @staticmethod
    def save(jet_list, outdir, filename):
        out_filename = os.path.join(outdir, filename + ".pkl")
        with open(out_filename, "wb") as f:
            pickle.dump(jet_list, f, protocol=2)


def dir2D(phi):
    return torch.tensor([np.sin(phi), np.cos(phi)])


def _traverse(root, delta_P=None, cut_off=None, rate=None):
    """
    This function call the recursive function _traverse_rec to make the trees starting from the root

    Inputs
    root: numpy array representing the initial jet momentum
    delta_P: Initial value for the parent mass squared
    cut_off: Min value of the mass squared below which evolution stops
    rate: parametrizes the exponential distribution
    M_hard: value for the mass of the jet (root of the binary tree)

    Outputs
    content: a list of numpy array representing the momenta flowing
        through every possible edge of the tree. content[0] is the root momentum
    tree: an array of integers >= -1, such that
        content[tree[2 * i]] and content[tree[2 * i + 1]] represent the momenta
        associated repsectively to the left and right child of content[i].
        If content[i] is a leaf, tree[2 * i] == tree[2 * i + 1] == 1
    deltas: mass squared value associated to content[i]
    draws: r value  associated to content[i]
    """

    tree = []
    content = []
    deltas = []
    draws = []

    leaves = []

    nodes = []

    def _traverse_rec(
        root, # 4d vector representing initial node
        parent_idx, # parent_indx
        is_left, # calculations of l and r differ
        delta_P = None,
        drew = None, # the last draw
        cut_off = None,
        rate = None, # rate for exponential distribution
    ):

        """
        Recursive function to make the jet tree.
        """
        nonlocal tree, content, deltas, draws, leaves, nodes
        # tree: array that represents a tree
        # content: a vector that hold each node
        # deltas: deltas[i] is the mass squared value of the node in content[i]
        # draws: rates of decay for each content
        # leaves: will be filled with the nodes that no longer make the mass squared cut-off

        idx = len(tree) // 2
        if parent_idx >= 0:
            if is_left:
                tree[2 * parent_idx] = idx
            else:
                tree[2 * parent_idx + 1] = idx

        # Insert 2 new nodes to the vector that constitutes the tree. 
        # In the next iteration we will replace this 2 values with the location of the parent of the new nodes
        tree.append(-1)
        tree.append(-1)

        # Fill the content vector with the values of the node, and the draws vector with the drawn mass squared value
        content.append(root) # make this into a dictionary
        draws.append(drew)

        # """ node """
        # currNode = 
        # nodes.append()
        # """ node """

        # checking if we should stop
        if delta_P > cut_off:
            deltas.append(delta_P)
        else:
            deltas.append(0)
            leaves.append(root)


        if delta_P > cut_off:

            # Sample uniformly over the sphere of unit radius a unit vector for the decay products in the CM frame
            r_CM = pyro.sample("rCM"+ str(idx) + str(is_left), multiNormal_dist)
            r_CM = r_CM.numpy()
            r_CM = r_CM / np.linalg.norm(r_CM)


            # Use different distributions to model the root node splitting, e.g. W decay
            if idx == 0:  sampling_dist = root_dist ## 2.1.1
            else: sampling_dist = decay_dist

            # Sample new values for the children invariant mass squared
            draw_decay_L = np.inf
            draw_decay_R = np.inf
            nL = 0
            nR = 0
            # print(f"draw_decay_L Before= {draw_decay_L, nL}")
            # print(f"draw_decay_R Before = {draw_decay_R, nR}")

            # The invariant mass squared should decrease strictly
            while draw_decay_L >= (1. - 1e-3): ## sample until value is smaller than that of parent
                draw_decay_L = pyro.sample(
                    "L_decay" + str(idx) + str(is_left), sampling_dist
                )  # We draw a number to get the left child delta
                nL += 1

            while draw_decay_R >= (1. - 1e-3):  ## sample until value is smaller than that of parent
                draw_decay_R = pyro.sample(
                    "R_decay" + str(idx) + str(is_left), sampling_dist
                )  # We draw a number to get the right child delta
                nR += 1

            t0 = delta_P
            tL = t0 * draw_decay_L ## eq (1)
            tR = (np.sqrt(t0) - np.sqrt(tL))**2 * draw_decay_R ## eq (2)

            if idx == 0:
                # print(f" Off-shell subjets mass = {np.sqrt(tL),np.sqrt(tR)}")
                pass

            # 2-Body decay in the parent CM frame
            EL_cm = CenterofMassE(tp = t0, t_child = tL, t_sib = tR)  ## eq (3)
            ER_cm = CenterofMassE(tp = t0, t_child = tR, t_sib = tL)

            P_CM = CenterofMassP(tp = t0, t_child = tR, t_sib = tL) ## eq (4), magnitude of 3-momentum

            # Boost to the lab frame, apply lorenz boost
            P0_lab = np.linalg.norm(root[1::])
            n0 = -root[1::]/P0_lab
            
            ## four dimensional vectors that characterize the two children that are being produced
            pL_mu = labEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = EL_cm, Pchild_CM = P_CM, p_versor = r_CM)
            pR_mu = labEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = ER_cm, Pchild_CM = P_CM, p_versor = -r_CM)
            
            """ node """
            node_pL_mu = nodeLabEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = EL_cm, Pchild_CM = P_CM, p_versor = r_CM)
            node_pR_mu = nodeLabEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = ER_cm, Pchild_CM = P_CM, p_versor = -r_CM)
            """ node """

            # Shuffle L and R randomly. This will contribute a factor of 1/2 to the likelihood
            flip = pyro.sample("Bernoulli" + str(idx), Bernoulli_dist)

            tL_rand = tR if flip == True else tL
            tR_rand = tL if flip == True else tR
            pL_mu_rand = pR_mu if flip == True else pL_mu
            pR_mu_rand = pL_mu if flip == True else pR_mu
            draw_decay_L_rand = draw_decay_R if flip == True else draw_decay_L
            draw_decay_R_rand = draw_decay_L if flip == True else draw_decay_R

            """ node """
            node_tL_rand = tR if flip == True else tL
            node_tR_rand = tL if flip == True else tR
            node_pL_mu_rand = node_pR_mu if flip == True else node_pL_mu
            node_pR_mu_rand = node_pL_mu if flip == True else node_pR_mu
            node_draw_decay_L_rand = draw_decay_R if flip == True else draw_decay_L
            node_draw_decay_R_rand = draw_decay_L if flip == True else draw_decay_R
            """ node """

            ## two recursive calls
            _traverse_rec(
                pL_mu_rand,
                idx,
                True,
                delta_P = tL_rand,
                cut_off = cut_off,
                rate = rate,
                drew = draw_decay_L_rand,
            )

            _traverse_rec(
                pR_mu_rand,
                idx,
                False,
                delta_P = tR_rand,
                cut_off = cut_off,
                rate = rate,
                drew = draw_decay_R_rand,
            )
    
    # Start from the root = jet 4-vector
    _traverse_rec(
        root,
        -1,
        False,
        delta_P=delta_P,
        cut_off=cut_off,
        rate=rate,
    )

    return tree, content, deltas, draws, leaves

### Auxiliary functions:
def CenterofMassE(tp = None,t_child = None,t_sib= None):
    # Decay product energies in the parent CM frame
    E = np.sqrt(tp)/2 * (1 + t_child/tp - t_sib/tp)
    return E

def CenterofMassP(tp= None, t_child= None, t_sib= None):
    # Decay product spatial momentum in the parent CM frame
    P = np.sqrt(tp)/2 * np.sqrt( 1 - 2 * (t_child+t_sib)/tp + (t_child - t_sib)**2 / tp**2 )

    return P

def labEP(tp= None,Ep_lab= None, Pp_lab= None , n= None, Echild_CM= None, Pchild_CM= None, p_versor= None):
    # Boost to the lab frame, function to apply lorenz boost and generate 4d vector for nodes
    
    # print(f"{type(tp)}")
    # print(f"{type(Ep_lab)}")

    tp = tp.numpy()
    Echild_CM = Echild_CM.numpy()
    Pchild_CM = Pchild_CM.numpy()

    Elab = Ep_lab/np.sqrt(tp)* Echild_CM - Pp_lab/np.sqrt(tp) * Pchild_CM * np.dot(n,p_versor)

    Plab = - Pp_lab/np.sqrt(tp) * Echild_CM * n + Pchild_CM * (p_versor + (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n)

    if Elab < np.linalg.norm(Plab):
        # print(f"---" * 10)
        # print(f" Elab = {Elab}")
        # print(f" Plab = {np.linalg.norm(Plab)}")
        # print(f" sqrt(tp) = {np.sqrt(tp)}")
        # print(f" Ep_lab = {Ep_lab}")
        # print(f" Pp_lab = {Pp_lab}")
        # print(f" np.dot(n,p_versor) = {np.dot(n,p_versor)}")
        # print(f"Echild CM = {Echild_CM}")
        # print(f"Pchild_CM = {Pchild_CM}")
        # print(f" terms = {Pp_lab/np.sqrt(tp) * Echild_CM * n,+ Pchild_CM * (p_versor ),Pchild_CM * ( (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n) }")
        # print(f"---" * 10)
        pass

    p_mu = np.concatenate(([Elab],Plab))
    return p_mu

##################################################################################################################

def nodeLabEP(tp= None,Ep_lab= None, Pp_lab= None , n= None, Echild_CM= None, Pchild_CM= None, p_versor= None):
    # Boost to the lab frame, function to apply lorenz boost and generate 4d vector for nodes

    tp = tp.numpy()
    Echild_CM = Echild_CM.numpy()
    Pchild_CM = Pchild_CM.numpy()

    Elab = Ep_lab/np.sqrt(tp)* Echild_CM - Pp_lab/np.sqrt(tp) * Pchild_CM * np.dot(n,p_versor)

    Plab = - Pp_lab/np.sqrt(tp) * Echild_CM * n + Pchild_CM * (p_versor + (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n)

    p_mu = np.concatenate(([Elab],Plab))
    currNodeVec4 = vec4(p_mu[0], p_mu[1], p_mu[2], p_mu[3])
    return currNodeVec4





