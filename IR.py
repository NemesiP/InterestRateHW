# -*- coding: utf-8 -*-
"""
Created on Sun May  2 17:10:58 2021

@author: peter
"""

import numpy as np
from abc import ABCMeta, abstractmethod

class InterestRateBase(object, metaclass = ABCMeta):
    def __init__(self, *args):
        self.args = args
        
    def _calculate_zero_coupon(self, forward_rates, dt, N):
        zero_coupon = np.zeros(N)
        
        for  i in range(N):
            if i == 0:
                zero_coupon[i] = np.exp( -forward_rates[i] * dt)
            else:
                zero_coupon[i] = zero_coupon[i - 1] * np.exp( - forward_rates[i] * dt)
        
        return zero_coupon
    
    @abstractmethod
    def _create_tree(self):
        pass
    
    @abstractmethod
    def _helper_tree(self):
        pass
    
    def _calculate_probabilities(self, helper_tree):
        pu, pd, pm = [], [], []
        for i in range(1, len(helper_tree) + 1):
            pu.append(1/6 + helper_tree[-i]**2 / 2 + helper_tree[-i] / 2)
            pm.append(2/3 - helper_tree[-i]**2)
            pd.append(1/6 + helper_tree[-i]**2 / 2 - helper_tree[-i] / 2)
            
        pu = pu[::-1]
        pm = pm[::-1]
        pd = pd[::-1]
        return pu, pm, pd
    
    def _create_ar_tree(self, r0, N, dt, interest_tree, helper_exp, zero_coupon, pu, pm, pd):
        AR = []
        phi = np.zeros(N)
           
        for i in range(N):
            if i == 0:
                AR.append(np.array([1.0]))
                phi[i] = (np.log(AR[i] * np.exp(-r0*dt))-np.log(zero_coupon[i]))/dt
            elif i == 1:
                AR.append(np.array([1 * pd[0] * np.exp(-(r0 + phi[0])*dt), 1 * pm[0] * np.exp(-(r0 + phi[0])*dt), 1 * pu[0] * np.exp(-(r0 + phi[0])*dt)]))
                phi[i] = (np.log(np.dot(helper_exp[i], AR[i])) - np.log(zero_coupon[i]))/dt
            else:
                arr = np.zeros(i * 2 + 1)
                for j in range(1, i*2 + 2):
                    if j == 1:
                        arr[j - 1] = AR[i - 1][0] * pd[i - 1][0] * np.exp(-(interest_tree[i - 1][0] + phi[i - 1]) * dt)
                    elif j == 2:
                        arr[j - 1] = (AR[i - 1][1] * pd[i - 1][1] * np.exp(-(interest_tree[i - 1][1] + phi[i - 1]) * dt) + 
                                      AR[i - 1][0] * pm[i - 1][0] * np.exp(-(interest_tree[i - 1][0] + phi[i - 1]) * dt))
                    elif j == i*2:
                        arr[j - 1] = (AR[i - 1][-1] * pm[i - 1][-1] * np.exp(-(interest_tree[i - 1][-1] + phi[i - 1]) * dt) + 
                                      AR[i - 1][-2] * pu[i - 1][-2] * np.exp(-(interest_tree[i - 1][-2] + phi[i - 1]) * dt))
                    elif j == i*2 + 1:
                        arr[j - 1] = AR[i - 1][-1] * pu[i - 1][-1] * np.exp(-(interest_tree[i - 1][-1] + phi[i - 1]) * dt)
                    else:
                        arr[j - 1] = (AR[i - 1][j - 3] * pu[i - 1][j - 3] * np.exp(-(interest_tree[i - 1][j - 3] + phi[i - 1]) * dt) + 
                                      AR[i - 1][j - 2] * pm[i - 1][j - 2] * np.exp(-(interest_tree[i - 1][j - 2] + phi[i - 1]) * dt) + 
                                      AR[i - 1][j - 1] * pd[i - 1][j - 1] * np.exp(-(interest_tree[i - 1][j - 1] + phi[i - 1]) * dt))
                AR.append(arr)
                phi[i] = (np.log(np.dot(helper_exp[i], AR[i])) - np.log(zero_coupon[i]))/dt
                
        return AR, phi
    
    def _create_phi_adjusted_tree(self, interest_tree, phi):
        interest_phi = []
        
        for i in range(len(phi)):
            interest_phi.append(interest_tree[i] + phi[i])
        
        return interest_phi
    
    def _calculate_pricing(self, interest_phi, K, forward_rates, dt, pu, pm, pd):
        pricing_tree = []
        
        for i in range(1, len(interest_phi) + 1):
            if i == 1:
                pricing_tree.append((interest_phi[-i] - K) * dt)
            elif i >= 3 and i <= 4:
                arr = np.zeros(interest_phi[-i].shape[0])
                for j in range(interest_phi[-i].shape[0]):
                    arr[j] = np.exp(-forward_rates[-i] * dt) * (pu[-i][j] *(pricing_tree[i - 2][j + 2] + dt *(interest_phi[-i + 1][j + 2] - K)) + pm[-i][j] *(pricing_tree[i - 2][j + 1] + dt *(interest_phi[-i + 1][j + 1] - K)) + pd[-i][j] *(pricing_tree[i - 2][j] + dt *(interest_phi[-i + 1][j] - K)))
                pricing_tree.append(arr)
            elif i >= 5 and i <= 11:
                arr = np.zeros(interest_phi[-i].shape[0])
                for j in range(interest_phi[-i].shape[0]):
                    arr[j] = max(np.exp(-forward_rates[-i] * dt) * (pu[-i][j] *(pricing_tree[i - 2][j + 2] + dt *(interest_phi[-i + 1][j + 2] - K)) + pm[-i][j] *(pricing_tree[i - 2][j + 1] + dt *(interest_phi[-i + 1][j + 1] - K)) + pd[-i][j] *(pricing_tree[i - 2][j] + dt *(interest_phi[-i + 1][j] - K))), 0)
                pricing_tree.append(arr)
            else:
                arr = np.zeros(interest_phi[-i].shape[0])
                for j in range(interest_phi[-i].shape[0]):
                    arr[j] = np.exp(-forward_rates[-i] * dt) * (pu[-i][j] * pricing_tree[i - 2][j + 2] + pm[-i][j] * pricing_tree[i - 2][j + 1] + pd[-i][j] * pricing_tree[i - 2][j])
                pricing_tree.append(arr)
        
        return pricing_tree
    
class VasicekModel(InterestRateBase):
    def __init__(self, kk, theta, sigma, r0, K, dt, N, forward_rates):
        self.kk = kk
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
        self.K = K
        self.dt = dt
        self.N = N
        self.dx = np.sqrt(3 * (self.sigma ** 2) / (2 * self.theta) * (1 - np.exp(-2 * self.theta * self.dt)))
        self.forward_rates = forward_rates
        
    def _create_tree(self):
        tree_mid = np.zeros(self.N + 1)
        interest_tree, helper_exp = [], []
        
        for i in range(self.N + 1):
            if i == 0:
                tree_mid[i] = self.r0
                interest_tree.append(np.array([tree_mid[i]]))
                helper_exp.append(np.exp(-interest_tree[i]*self.dt))
            else:
                tree_mid[i] = tree_mid[i - 1] * np.exp(-self.theta*self.dt) + (self.kk/self.theta) * (1 - np.exp(-self.theta * self.dt))
#                If you allow to the interest rates can take negative values use the commented code
#                interest_tree.append(tree_mid[i] + self.dx * np.linspace(-i, i, i*2 + 1))
#
#                You this instead to get only positive values in the interest rate tree
                interest_tree.append(np.abs(tree_mid[i] + self.dx * np.linspace(-i, i, i*2 + 1)))
                helper_exp.append(np.exp(-interest_tree[i]*self.dt))
        
        return interest_tree, helper_exp
    
    def _helper_tree(self, interest_tree):
        helper_tree = []
        
        for i in range(1, len(interest_tree)):
            m = interest_tree[-i][1:-1]
            p = interest_tree[-i-1] * np.exp(-self.theta * self.dt) + (self.kk / self.theta) * (1 - np.exp( - self.theta * self.dt))
            helper_tree.append((p - m )/ self.dx)
            
        helper_tree = helper_tree[::-1]
        
        return helper_tree
    
    def price(self):
        self.interest_tree, self.helper_exp = self._create_tree()
        self.helper_tree = self._helper_tree(self.interest_tree)
        self.pu, self.pm, self.pd = self._calculate_probabilities(self.helper_tree)
        self.zero_coupon = self._calculate_zero_coupon(self.forward_rates, self.dt, self.N)
        self.AR, self.phi = self._create_ar_tree(self.r0, self.N, self.dt, self.interest_tree, self.helper_exp, self.zero_coupon, self.pu, self.pm, self.pd)
        self.interest_phi = self._create_phi_adjusted_tree(self.interest_tree, self.phi)
        self.pricing = self._calculate_pricing(self.interest_phi, self.K, self.forward_rates, self.dt, self.pu, self.pm, self.pd)
        return self.pricing[-1][0]
    
class CIRModel(InterestRateBase):
    def __init__(self, alpha, beta, sigma, r0, K, dt, N, forward_rates):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.r0 = r0
        self.K = K
        self.dt = dt
        self.N = N
        self.dx = np.sqrt(3 * self.dt * (self.sigma/2)**2)
        self.forward_rates = forward_rates
        
    def _create_tree(self):
        tree_mid = np.zeros(self.N + 1)
        interest_tree, helper_exp = [], []
        
        for i in range(self.N + 1):
            if i == 0:
                tree_mid[i] = np.sqrt(self.r0)
                interest_tree.append(np.array([tree_mid[i]]))
                helper_exp.append(np.exp(-interest_tree[i]*self.dt))
            else:
                tree_mid[i] = tree_mid[i - 1] + self.dt * ((self.alpha / 2 - (self.sigma**2) / 8) / tree_mid[i - 1] - self.beta / 2 * tree_mid[i - 1])
#                If you allow to the interest rates can take negative values use the commented code
#                interest_tree.append(tree_mid[i] + self.dx * np.linspace(-i, i, i*2 + 1))
#
#                You this instead to get only positive values in the interest rate tree
                interest_tree.append(np.abs(tree_mid[i] + self.dx * np.linspace(-i, i, i*2 + 1)))
                helper_exp.append(np.exp(-(interest_tree[i]**2)*self.dt))
        
        return interest_tree, helper_exp, tree_mid
    
    def _helper_tree(self, interest_tree, tree_mid):
        helper_tree = []
        
        for j in range(1, self.N + 1):
            m = interest_tree[-j][1:-1]
            sem = tree_mid[-j-1] + self.dx * np.linspace(- self.N + j, self.N - j, (self.N - j) * 2 + 1)
            sem[abs(sem) <= 0.04] = np.sign(sem[abs(sem) <= 0.04]) * np.sqrt(abs(sem[abs(sem) <= 0.04]))
            p = interest_tree[-j-1] + self.dt * ((self.alpha / 2 - (self.sigma**2) / 8) / sem - self.beta / 2 * interest_tree[-j-1])
            helper_tree.append((p - m )/ self.dx) 
        
        helper_tree = helper_tree[::-1]
        
        return helper_tree
    
    def _calc_interest_tree_square(self, interest_tree):
        interest_tree_s = []
        for i in interest_tree:
            interest_tree_s.append(i ** 2)
        
        return interest_tree_s
    
    def price(self):
        self.interest_tree, self.helper_exp, tree_mid = self._create_tree()
        self.helper_tree = self._helper_tree(self.interest_tree, tree_mid)
        self.pu, self.pm, self.pd = self._calculate_probabilities(self.helper_tree)
        self.zero_coupon = self._calculate_zero_coupon(self.forward_rates, self.dt, self.N)
        self.interest_tree_2 = self._calc_interest_tree_square(self.interest_tree)
        self.AR, self.phi = self._create_ar_tree(self.r0, self.N, self.dt, self.interest_tree_2, self.helper_exp, self.zero_coupon, self.pu, self.pm, self.pd)
        self.interest_phi = self._create_phi_adjusted_tree(self.interest_tree_2, self.phi)
        self.pricing = self._calculate_pricing(self.interest_phi, self.K, self.forward_rates, self.dt, self.pu, self.pm, self.pd)
        return self.pricing[-1][0]
    
    
if __name__ == '__main__':
    frates = np.array([0.046, 0.05, 0.053, 0.046, 0.041, 0.045, 0.0475, 0.05, 0.051, 0.045, 0.045, 0.048, 0.0475, 0.047])
    vasi_p = VasicekModel(kk = 0.0045, theta = 0.1, sigma = 0.01, r0 = 0.046, K = 0.047, dt = 0.5, N = 14, forward_rates = frates)
    print(vasi_p.price())
    cir_p = CIRModel(alpha = 0.0045, beta = 0.1, sigma = 0.0447, r0 = 0.046, K = 0.047, dt = 0.5, N = 14, forward_rates = frates)
    print(cir_p.price())