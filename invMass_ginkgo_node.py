#!/usr/bin/env python

import numpy as np
import time
import sys
import os
import copy
import pickle
import torch
import collections
from torch import nn
import pyro
from ginkgo.pyro_simulator import PyroSimulator
# from ginkgo import likelihood_invM as likelihood
# from ginkgo import auxFunctions
from node import *
from likelihood_node import *
from pptree import *

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
        self.ret_root_list = []
    def bfs(self, root):
        nodeCount = 0
        q = collections.deque([[nodeCount, root]])
        pRootNode = Node(str(0))
        printQ = collections.deque([pRootNode])
        while q:
            nodeID, currNode = q.popleft()
            currPrintNode = printQ.popleft()

            print("Node", nodeID)
            print(" Vec4:", currNode.vec4)
            print(" Decay Rate:", currNode.decay_rate)
            print(" Mass Squared:", currNode.delta)
            print(" Log Likelihood:", currNode.logLH)
            print(" DIJ List:", *currNode.dijList[:5])
            print()
            
            self.ret_root_list.append(currNode)

            if currNode.left:
                nodeCount += 1
                q.append([nodeCount, currNode.left])

                printLeftNode = Node(str(nodeCount), currPrintNode)
                printQ.append(printLeftNode)

            if currNode.right:
                nodeCount += 1
                q.append([nodeCount, currNode.right])

                printRightNode = Node(str(nodeCount), currPrintNode)
                printQ.append(printRightNode)

        return pRootNode
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

        i = 0
        root_list = []
        # while len(root_list) < self.num_samples and i < self.maxNTry:
        while len(root_list) < self.maxNTry:

            # generate a tree
            currNode = _traverse(
                self.jet_p,
                delta_P=self.Delta_0,
                cut_off=self.pt_cut,
                rate=decay_rate,
            )
            currNode = enrich_jet_logLH(rootNode = currNode, delta_min = self.pt_cut, dij = True)
            root_list.append(currNode)

        for i in root_list:
            print_tree(self.bfs(i))
        return self.ret_root_list

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
    CUT_OFF = cut_off
    def _calc(root, delta_P):
        nonlocal is_root

        # Sample uniformly over the sphere of unit radius a unit vector for the decay products in the CM frame
        r_CM = pyro.sample("rCM", multiNormal_dist)
        r_CM = r_CM.numpy()
        r_CM = r_CM / np.linalg.norm(r_CM)


        # Use different distributions to model the root node splitting, e.g. W decay
        if is_root:  sampling_dist = root_dist ## 2.1.1
        else: sampling_dist = decay_dist

        # Sample new values for the children invariant mass squared
        draw_decay_L = np.inf
        draw_decay_R = np.inf
        nL = 0
        nR = 0

        # The invariant mass squared should decrease strictly
        while draw_decay_L >= (1. - 1e-3): ## sample until value is smaller than that of parent
            draw_decay_L = pyro.sample(
                "L_decay", sampling_dist
            )  # We draw a number to get the left child delta
            nL += 1

        while draw_decay_R >= (1. - 1e-3):  ## sample until value is smaller than that of parent
            draw_decay_R = pyro.sample(
                "R_decay", sampling_dist
            )  # We draw a number to get the right child delta
            nR += 1

        t0 = delta_P
        tL = t0 * draw_decay_L ## eq (1)
        tR = (np.sqrt(t0) - np.sqrt(tL))**2 * draw_decay_R ## eq (2)

        # 2-Body decay in the parent CM frame
        EL_cm = CenterofMassE(tp = t0, t_child = tL, t_sib = tR)  ## eq (3)
        ER_cm = CenterofMassE(tp = t0, t_child = tR, t_sib = tL)

        P_CM = CenterofMassP(tp = t0, t_child = tR, t_sib = tL) ## eq (4), magnitude of 3-momentum

        # Boost to the lab frame, apply lorenz boost
        P0_lab = np.linalg.norm(root[1::])
        n0 = -root[1::]/P0_lab
        
        node_pL_mu = nodeLabEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = EL_cm, Pchild_CM = P_CM, p_versor = r_CM)
        node_pR_mu = nodeLabEP(tp = t0, Ep_lab = root[0], Pp_lab = P0_lab, n = n0, Echild_CM = ER_cm, Pchild_CM = P_CM, p_versor = -r_CM)

        return tL, tR, node_pL_mu, node_pR_mu, draw_decay_L, draw_decay_R

    def _traverse_rec(currNode):
        """
        Recursive function to make the jet tree.
        """
        nonlocal CUT_OFF, is_root
        if currNode.delta < CUT_OFF:
            return

        root = currNode.vec4 # 4d vector representing initial node


        delta_P = currNode.delta # invar mass sqrd of last draw
        drew = currNode.decay_rate # decay rate of last draw

        tL, tR, node_pL_mu, node_pR_mu, draw_decay_L, draw_decay_R = _calc(root, delta_P)

        if is_root:
            is_root = False

        # Shuffle L and R randomly. This will contribute a factor of 1/2 to the likelihood
        flip = pyro.sample("Bernoulli", Bernoulli_dist)
        flip = flip > 0.5
        # print(flip)
        node_tL_rand = tR if flip == True else tL
        node_tR_rand = tL if flip == True else tR
        node_pL_mu_rand = node_pR_mu if flip == True else node_pL_mu
        node_pR_mu_rand = node_pL_mu if flip == True else node_pR_mu
        node_draw_decay_L_rand = draw_decay_R if flip == True else draw_decay_L
        node_draw_decay_R_rand = draw_decay_L if flip == True else draw_decay_R
        
        leftNode = jetNode(node_pL_mu_rand, None, None, node_draw_decay_L_rand, node_tL_rand)
        currNode.left = leftNode
        _traverse_rec(leftNode)

        rightNode = jetNode(node_pR_mu_rand, None, None, node_draw_decay_R_rand, node_tR_rand)
        currNode.right = rightNode
        _traverse_rec(rightNode)
        
    rootNode = jetNode(vec4 = root, left = None, right = None, decay_rate = 1, delta = delta_P)
    is_root = True
    # Start from the root = jet 4-vector
    _traverse_rec(rootNode)
    return rootNode

### Auxiliary functions:
def CenterofMassE(tp = None,t_child = None,t_sib= None):
    # Decay product energies in the parent CM frame
    E = np.sqrt(tp)/2 * (1 + t_child/tp - t_sib/tp)
    return E

def CenterofMassP(tp= None, t_child= None, t_sib= None):
    # Decay product spatial momentum in the parent CM frame
    P = np.sqrt(tp)/2 * np.sqrt( 1 - 2 * (t_child+t_sib)/tp + (t_child - t_sib)**2 / tp**2 )

    return P
    
def nodeLabEP(tp= None,Ep_lab= None, Pp_lab= None , n= None, Echild_CM= None, Pchild_CM= None, p_versor= None):
    # Boost to the lab frame, function to apply lorenz boost and generate 4d vector for nodes

    tp = tp.numpy()
    Echild_CM = Echild_CM.numpy()
    Pchild_CM = Pchild_CM.numpy()

    Elab = Ep_lab/np.sqrt(tp)* Echild_CM - Pp_lab/np.sqrt(tp) * Pchild_CM * np.dot(n,p_versor)

    Plab = - Pp_lab/np.sqrt(tp) * Echild_CM * n + Pchild_CM * (p_versor + (Ep_lab/np.sqrt(tp) - 1) * np.dot(p_versor,n) * n)

    p_mu = np.concatenate(([Elab],Plab))
    return p_mu





