"""
Copyright (c) 2021 Chris van der Heide and Liam Hodgkinson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
import pymc3
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import scipy
torch.set_default_dtype(torch.float64)

def scipy_lbfgs(fun, jac, x0):
    """A wrapper around the SciPy implementation of L-BFGS-B"""
    result = scipy.optimize.minimize(fun, x0, jac=jac, method='L-BFGS-B')
    if not result['success']:
        raise RuntimeError("L-BFGS-B failed to converge")
    return result['x']



class ShadowTorch(object):
    def __init__(self, initial_parameters, samples = 100, potential = 'blr',
                 integrator = 'HMC', shadow = False, metric = 'Euclidean',
                 momentum_retention = 0.,  dataset=None, labels = None, stepsize=.1, 
                 prior_var=100.0, max_leapfrog_steps=6, max_fixed_point_iterations = 100,
                 fixed_point_threshold = 1e-3, softabs_constant = 1e6, max_shadow = None,
                 verbose = False, jitter = 5e-3, prior = 'Gaussian', trajectories = False,
                 random_retention = False, store_paths = False):
        self.parameters = initial_parameters.requires_grad_(True)
        self.momentum = torch.zeros_like(self.parameters)
        self.n_samples = samples
        self._potential = potential
        self._potentials = ['blr', 'banana', 'funnel']
        self._priors = ['Gaussian', 'Cauchy', 'Sparse']
        self._prior = prior
        self.stepsize = stepsize
        self.prior_var = prior_var
        self._max_leapfrog_steps = max_leapfrog_steps
        self._max_fixed_point_iterations = max_fixed_point_iterations
        self._fixed_point_threshold = fixed_point_threshold
        self.samples = [self.parameters.detach()]
        self.momenta = []
        self.rejected = 0
        self.accepted = 0
        self._integrators = ['HMC', 'RMHMC']
        self._integrator = integrator
        self.momentum_retention = momentum_retention
        self.shadow = shadow
        self.shadow_ = torch.zeros(1)
        self._metrics = ['Euclidean', 'Hessian', 'Banana', 'Softabs', 'Fisher', 'SoftDiag']
        self._metric = metric
        self.softabs = softabs_constant
        self.dataset = dataset
        self.labels = labels
        self.hamiltonians = []
        self.shadows = []
        self.radon_nikodym = []
        self.hamiltonian_error = []
        self.shadow_hamiltonian_error = []
        self.momentum_accepts_ = []
        self.momentum_accepted_ = []
        self.rands_ = []
        self.paths = []
        self.metric_ = torch.zeros(initial_parameters.shape[0],initial_parameters.shape[0])
        self.inverse_ = torch.zeros_like(self.metric_)
        self.cholesky_inv_ = torch.zeros_like(self.metric_)
        self.potential_ = torch.zeros(1)
        self.capacitor_ = torch.zeros(1)
        self.kinetic_ = torch.zeros(1)
        self.hamiltonian_ = torch.zeros(1)
        self.verbose = verbose
        self.momentum_diverged = 0
        self.parameters_diverged = 0
        self.jitter = jitter
        self.jitters = torch.zeros_like(self.parameters)
        self.elapsed = 0
        self.trajectories = trajectories
        self._cov = torch.zeros(1)
        self.random_retention = random_retention
        self.degenerate = False
        self.degenerates = 0
        self.completed = False
        self.store_paths = store_paths
        self.max_shadow = max_shadow
        
    def get_cross_entropy(self):
        """Returns U(q), the negative log-posterior formed from Bayesian Logistic Regression"""
        assert (self.dataset is not None) and (self.labels is not None), 'Logistic Regression requires a dataset and labels.'
        potential = 0.0
        logits = self.dataset @ self.parameters[:self.dataset.shape[1]]
        max_logits = torch.max(torch.zeros(logits.shape[0]),logits)
        potential = (-logits @ self.labels.t() + torch.sum(max_logits) + torch.sum(
            torch.log(torch.exp(-max_logits)+torch.exp(logits - max_logits))))# * n.reciprocal())
        return potential
    
    def get_prior(self):
        """"Returns the negative log of the specified prior distribution"""
        assert self._prior in self._priors, 'Unsupported prior! Check the _priors attribute for a list of priors.'
        if self._prior == 'Gaussian':
            prior = 0.5 * torch.sum(self.parameters ** 2)/self.prior_var
        elif self._prior == 'Cauchy':
            dimconst = (self.parameters.shape[0] + 1)/2.
            prior = dimconst*torch.log(self.prior_var + torch.sum(self.parameters ** 2))
        elif self._prior == 'Sparse':
            n = self.dataset.shape[1]
            gauss_prior = 0.5 * torch.sum(torch.exp(self.parameters[-1] * torch.exp(self.parameters[n:2*n]) * self.parameters[:n] ** 2))
            gamma_density = torch.distributions.Gamma(1.5,0.5)
#            gamma_prior = -gamma_density.log_prob(torch.exp(self.parameters[n:])).sum()
#            lambda_density = torch.distributions.Gamma(1.5,0.5)
            lambda_prior = -gamma_density.log_prob(torch.exp(self.parameters[n:])).sum()
            prior = gauss_prior + lambda_prior
        return prior
        
    def get_banana(self):
         """Returns U(q), the negative log-posterior formed from the banana distribution"""  
         assert self.parameters.shape[0] == 2, 'The banana distribution requires two dimensional parameters'
         datashift = self.parameters[0] + self.parameters[1]**2 - self.dataset
         logbanana = 0.5*torch.sum(datashift**2)/self.prior_var + 0.5*(self.parameters[0]**2 + self.parameters[1]**2)
         return logbanana.squeeze()
        
    def get_funnel(self):
         """Returns U(q), the negative log-posterior formed from Neal's funnel distribution
         of the same dimension as the initial parameters"""
         v_density = torch.distributions.Normal(0,3)
         potential1 = -v_density.log_prob(self.parameters[0])
         x_density = torch.distributions.Normal(0,torch.exp(-self.parameters[0])**0.5)
         potential2 = -x_density.log_prob(self.parameters[1:]).sum()
         return potential1 + potential2

    def get_potential(self):
        """Returns U(q), the negative log-posterior formed from the prescribed potential function"""
        assert self._potential in self._potentials, 'Unsupported potnetial function! Check the _potentials attribute for a list of supported potentials.'
        if self._potential == 'blr':
            log_likelihood = self.get_cross_entropy()
            prior = self.get_prior()
            potential = log_likelihood + prior
        if self._potential == 'banana':
            potential = self.get_banana()
        if self._potential == 'funnel':
            potential = self.get_funnel()
        return potential

    def get_metric(self):
        """Returns the metric prescribed in self._metric, must be in the supported list given by self._metrics"""
        assert self._metric in self._metrics, 'Unsupported metric! Check the _metrics attribute for a list of supported metrics.'
        if self._metric == 'Euclidean':
            metric = torch.eye(self.parameters.shape[0])
        elif self._metric == 'Banana':
            n = self.dataset.shape[0]
            fish = torch.zeros(2,2)
            fish[0,0] = n/self.prior_var + 1
            fish[0,1] = n*2*self.parameters[1]/self.prior_var
            fish[1,0] = n*2*self.parameters[1]/self.prior_var
            fish[1,1] = n*4*self.parameters[1]**2/self.prior_var + 1
            metric = fish
        elif self._metric == 'Hessian':
            metric = self.get_hessian()
        elif self._metric == 'Softabs':
            hessian = self.get_hessian()
            if self._potential == 'funnel':
                hessian += torch.diag(self.jitters)
            eigs, vects = hessian.symeig(eigenvectors = True)
            softabs = (1./torch.tanh(self.softabs * eigs)) * eigs
            metric = vects @ softabs.diag() @ vects.t()
        elif self._metric == 'Fisher':
            metric = torch.zeros(self.parameters.shape[0],self.parameters.shape[0])
            grads = torch.zeros(self.parameters.shape[0])
            grads[0] = 0.5*torch.sum(self.parameters[1:]**2)*torch.exp(self.parameters[0]) + self.parameters[0]/9.
            grads[1:] = self.parameters[1:]*torch.exp(self.parameters[0])
            metric = torch.ger(grads,grads) + torch.eye(self.parameters.shape[0])/self.softabs
        return metric
        
    def get_hamiltonian(self):
        """Returns the Hamiltonian H(q,p) = U(q) + C(q) + K(q,p), where C is the 
        position-dependent part of the normalizing constant"""
        assert (self._integrator == 'HMC' and self._metric == 'Euclidean') or self._integrator == 'RMHMC', 'Parameter dependent metrics require the RMHMC integrator'
        if self._integrator == 'RMHMC':# and self._metric != 'Euclidean':
            self.potential_ = self.get_potential()
            self.metric_ = self.get_metric()
            self.inverse_ = self.metric_.inverse()
            self.capacitor_ = self.get_capacitor()
            self.kinetic_ = self.get_kinetic()
            ham = self.potential_ + self.capacitor_ + self.kinetic_
        else:
            self.potential_ = self.get_potential()
            self.kinetic_ = self.get_kinetic()
            ham = self.potential_ + self.kinetic_
        self.hamiltonian_ = ham
        return ham
    
    def get_capacitor(self):
        """Returns the position-dependent part C(q) of the normalizing constant
        of the conditional distribution K(q,p)"""
        cap = 0.5 * self.metric_.logdet()
        return cap
    
    def get_kinetic(self):
        if self._metric == 'Euclidean':
            kinetic = 0.5*torch.sum(self.momentum**2)
        else:
            kinetic = 0.5 * self.momentum @ self.inverse_ @ self.momentum.t()
        return kinetic
            
    def get_hessian(self):
        '''
        Compute the Hessian of the potential with respect to the parameters
        '''
        if self._potential == 'blr' and self._prior == 'Gaussian':
            logits = self.dataset @ self.parameters
            el = torch.exp(logits)
            probs = el/(1. + el)
            pf = probs - probs**2
            px = pf*self.dataset.t()
            fish = px @ self.dataset
            hess = fish + torch.eye(self.parameters.shape[0])/self.prior_var
        elif self._potential == 'funnel':
            hess = torch.zeros(self.parameters.numel(),self.parameters.numel())
            hess[0,0] = 0.5*torch.sum(self.parameters[1:]**2)*torch.exp(self.parameters[0]) + 1/9.
            hess[1:,0] = self.parameters[1:]*torch.exp(self.parameters[0])
            hess[0,1:] = hess[1:,0]
            hess[1:,1:] = torch.exp(self.parameters[0]) * torch.eye(self.parameters.shape[0]-1)
        elif self._prior == 'Sparse':
            rate = 1.5
            n = self.dataset.shape[1]
            cehess = torch.zeros(n,n)
            logits = logits = self.dataset @ self.parameters[:n]
            el = torch.exp(logits)
            probs = el/(1. + el)
            pf = probs - probs**2
            px = pf*self.dataset.t()
            fish = px @ self.dataset
            hess = torch.zeros(self.parameters.numel(),self.parameters.numel())
            sigmas = torch.exp(self.parameters[n:2*n])
            tau = torch.exp(self.parameters[-1])
            hess[:n,:n] = cehess + torch.diag(tau * sigmas)
            vec = self.parameters[:n] * sigmas * tau
            cross = torch.diag(vec)
            hess[:n,n:2*n] = cross
            hess[n:2*n,:n] = cross
            hess[:n,-1] = vec
            hess[-1,:n] = vec
            hess[n:2*n,n:2*n] = torch.diag(vec + rate * sigmas)
            hess[-1,-1] = torch.sum(0.5 * tau * sigmas * self.parameters[:n]**2) + rate * tau
        else:
            gradient = torch.autograd.grad(self.potential_, self.parameters, create_graph=True, allow_unused=False)[0]
            hess = torch.zeros(self.parameters.numel(),self.parameters.numel())
            for row in range(self.parameters.numel()):
                hess[row] = torch.autograd.grad(gradient[row], self.parameters, create_graph=True)[0]
                del gradient
        return hess

    def resample_momenta(self, init = False):
        if (self.shadow == False) or init:
            if self._integrator == 'HMC':
                momentum = torch.distributions.Normal(torch.zeros_like(
                    self.parameters),  torch.ones_like(self.parameters)).sample()
            else:
                new_momentum = torch.distributions.MultivariateNormal(torch.zeros_like(
                    self.parameters), self.metric_).sample()
                momentum = torch.sqrt(torch.tensor(self.momentum_retention)
                                           ) * self.momentum + torch.sqrt(
                                               1 -  torch.tensor(self.momentum_retention)) * new_momentum
        if (self.shadow == True) and (self._integrator == "HMC"):
            return None
            #to be implemented
        if (self.shadow == True) and (self._integrator == "RMHMC"):
            momentum_sample = torch.distributions.MultivariateNormal(torch.zeros_like(
                self.parameters), self.metric_).sample().detach()
            self.parameters = self.parameters.detach().requires_grad_(True)
            self.hamiltonian_ = self.get_hamiltonian()
            #noisey_kinetic = 0.5 * momentum_sample @ self.inverse_ @ momentum_sample.t()
            if self.random_retention:
                energy_ratio = torch.rand(1)       
            else:    
                energy_ratio = torch.tensor(1.)#*torch.max(noisey_kinetic/self.kinetic_, torch.tensor(0.99))
            proposed_momentum = torch.sqrt(energy_ratio * self.momentum_retention**2
                                           ) * self.momentum + torch.sqrt(
                                               1 - energy_ratio * self.momentum_retention**2) * momentum_sample
            proposed_kinetic = 0.5 * proposed_momentum @ self.inverse_ @ proposed_momentum.t()
            proposed_hamiltonian = self.hamiltonian_ + proposed_kinetic - self.kinetic_
            #this stuff wont work
            self.shadow_ = self.get_shadow()
            #change attributes here
            stored_momentum = self.momentum.clone()
            stored_hamiltonian = self.hamiltonian_.clone()
            self.hamiltonian_ = proposed_hamiltonian
            self.momentum = proposed_momentum
            proposed_shadow = self.get_shadow()
            momentum_accept = self.shadow_ - proposed_shadow# - self.kinetic_ + proposed_kinetic
            lograndom = torch.rand(1).log()
            accepted = 0
            if momentum_accept > lograndom:
                accepted = 1
                self.shadow_ = proposed_shadow
                self.kinetic_ = proposed_kinetic
            else:
                self.hamiltonian_ = stored_hamiltonian
                self.momentum = stored_momentum
#            print(accepted)
            momentum = self.momentum
            self.momentum_accepts_.append(momentum_accept.detach())
            self.momentum_accepted_.append(accepted)
        return momentum

    def get_shadow(self):
        """Calculated the second order terms of the shadow Hamiltonian.
        requires a precomputed self.hamiltonian_"""
        grad_H = torch.autograd.grad(self.hamiltonian_,self.parameters,create_graph = True)[0]
        grad_H_copy = grad_H.clone().detach()
        velocity = self.inverse_ @ self.momentum.t()
        velocity_copy = velocity.clone().detach()
        prehessian = velocity_copy @ grad_H.t()
        hessian_velocity = torch.autograd.grad(prehessian,self.parameters,retain_graph = True)[0]
        shadow1 = hessian_velocity @ velocity.t()
        premix = velocity @ grad_H_copy.t()
        mixed_term = torch.autograd.grad(premix,self.parameters,retain_graph = True)[0]
        shadow2 = -0.5 * grad_H @ self.inverse_ @ grad_H.t()
        shadow3 = mixed_term @ velocity.t()
        shadow = self.stepsize**2 * (shadow1 + shadow2 + shadow3)/12
        return shadow

    def _is_nanned(self, input):
        if torch.is_tensor(input):
            input = torch.sum(input)
            isnan = int(torch.isnan(input)) > 0
            isinf = int(torch.isinf(input)) > 0
            return isnan or isinf
        else:
            input = float(input)
            return (input == float('inf')) or (input == float('-inf')) or (input == float('NaN'))

    def implicit_half_kick(self):
        """Applies the implicit momentum update
        p -> p_0 - 1/2 * delta * dH/dq(q,p)"""
        stored_momentum = self.momentum.clone()
        rejected = False
        best_diff = torch.tensor(100.)
        for iter in range(self._max_fixed_point_iterations):
            old_momentum = self.momentum.clone()
            self.parameters = self.parameters.detach().requires_grad_(True)
            hamiltonian = self.get_hamiltonian()
            d_hamiltonian = torch.autograd.grad(hamiltonian,self.parameters,retain_graph = True)[0]
            self.momentum = stored_momentum - 0.5 * self.stepsize * d_hamiltonian
            if self._is_nanned(self.momentum):
                rejected = True
                self.momentum = old_momentum
                if self.verbose:
                    print('Nanned during momentum update {}'.format(iter))
                break
            diff = torch.max((old_momentum - self.momentum) ** 2)
            if diff < self._fixed_point_threshold:
                break
            if diff<best_diff:
                best_diff = diff
            elif iter == self._max_fixed_point_iterations-1:
                rejected = True
                if self.verbose:
                    print('Exceeded maximum iterations during momentum update')
                break
#                print('Warning: reached {} iterations in momentum update. Smallest iteration ({}) was selected'.format(self._max_fixed_point_iterations, best_diff.item()))
        momentum = self.momentum
        return momentum, rejected
    
    def implicit_drift(self):
        """Applies the implicit momentum update
        q -> q_0 + 1/2 * delta * [d_H/dp(q_0,p) + dH/dp(q,p)]"""
        stored_parameters = self.parameters.detach().clone()
        self.potential_ = self.get_potential()
        old_inverse_metric = self.get_metric().inverse()
        old_drift = self.momentum @ old_inverse_metric
        best_drift = self.parameters.clone().detach()
        best_diff = torch.tensor(100.)
        rejected = False
        for iter in range(self._max_fixed_point_iterations):
            if self._is_nanned(self.parameters):
                rejected = True
                self.parameters = stored_parameters
                if self.verbose:
                    print('Nanned during parameter update')
                break
            previous_parameters = self.parameters.detach().clone()
            self.potential_ = self.get_potential()
            new_metric_inverse = self.get_metric().inverse()
            new_drift =  self.momentum @ new_metric_inverse
            self.parameters = stored_parameters + 0.5 * self.stepsize * (old_drift + new_drift)
            diff = torch.max((previous_parameters - self.parameters) ** 2)
            if diff < self._fixed_point_threshold:
                break
            if diff<best_diff:
                best_diff = diff
                best_drift = self.parameters              
            elif iter == self._max_fixed_point_iterations-1:
                self.parameters = best_drift
                rejected = True
                if self.verbose:
                    print('Exceeded maximum iterations during parameter update')
                break
        params = self.parameters
        return params, rejected

    def explicit_drift(self):
        """Applies the explicit momentum update q -> q + delta * dH/dp"""
        params = self.parameters + self.stepsize * self.momentum
        return params.detach()
    
    def explicit_half_kick(self):
        """Applies the explicit momentum update p -> p - 1/2 * delta * dH/dq"""
        rejected = False
        momentum = self.momentum.clone()
        old_momentum = momentum.clone()
        hamiltonian = self.get_hamiltonian()
        d_hamiltonian = torch.autograd.grad(hamiltonian, self.parameters, retain_graph=False)[0]
        momentum -= 0.5 * self.stepsize * d_hamiltonian
        if self._is_nanned(self.momentum):
            rejected = True
            self.momentum = old_momentum
#            if self.verbose:
            print('Nanned during explit momentum update {}'.format(iter))
        return momentum, rejected

    def step(self):
        """Draws a single HMC sample"""
        if self.store_paths:
            leapfrog_steps = self._max_leapfrog_steps
        else:
            leapfrog_steps = torch.ceil(self._max_leapfrog_steps * torch.rand(1)).int()
        self.potential_ = self.get_potential()
        self.metric_ = self.get_metric()
        self.momentum = self.resample_momenta()
        self.hamiltonian_ = self.get_hamiltonian()
        old_hamiltonian = self.hamiltonian_
        if self.shadow:
            if self.max_shadow is not None:
                old_shadow = torch.max(self.shadow_.clone() + self.max_shadow, old_hamiltonian)
            else:
                old_shadow = self.shadow_.clone()
        rejected = False
        for step in range(leapfrog_steps):
            if (self._integrator == 'RMHMC') and (self.lbfgs == False):
                self.momentum, rejected = self.implicit_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
                self.parameters, rejected = self.implicit_drift()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
                self.momentum, rejected = self.explicit_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
            elif self.lbfgs == True:
                self.momentum, rejected = self.lbfgs_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
                self.parameters, rejected = self.lbfgs_drift()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
                self.momentum = self.explicit_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
                if rejected:
                    break
            else:
                self.momentum, rejected = self.explicit_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
                self.parameters = self.explicit_drift()
                self.parameters = self.parameters.detach().requires_grad_(True)
                self.momentum, rejected = self.explicit_half_kick()
                self.parameters = self.parameters.detach().requires_grad_(True)
            if self.store_paths:
                self.paths.append(self.parameters.detach())
        new_hamiltonian = self.get_hamiltonian()
        ratio = old_hamiltonian - new_hamiltonian
        self.hamiltonian_error.append(ratio.detach().unsqueeze(0))
        if self.shadow:
            if self.max_shadow is not None:
                new_shadow = torch.max(self.get_shadow() + self.max_shadow, new_hamiltonian)
            else:
                new_shadow = self.get_shadow()
            shadow_error = old_shadow - new_shadow
            newratio = ratio + shadow_error
            self.shadow_hamiltonian_error.append(newratio.detach().unsqueeze(0))
            ratio = newratio

        uniform_rand = torch.rand(1)
        if uniform_rand >= torch.exp(ratio):
            # Reject sample
            rejected = True

        if rejected:
            if (len(self.momenta) > 10) and (self.momenta[-1] == self.momenta[-10]).sum().item():
                self.degenerate = True
            self.rejected += 1
            self.momentum = self.momenta[-1]
            self.parameters = self.samples[-1].clone().detach().requires_grad_(True)
            if self.shadow:
                radon_nikodym = torch.exp(old_shadow).unsqueeze(0)
            
            if self.verbose:
                print("(Rejected)", int(self.acceptance_rate() * 100), "%; Log-ratio: ",
                      ratio.detach())
        else:
            self.accepted += 1
            if self.shadow:
                radon_nikodym = torch.exp(new_shadow).unsqueeze(0)
            if self.verbose:
                print("(Accepted)", int(self.acceptance_rate() * 100), "%; Log-ratio: ",
                      ratio.detach())
        self.samples.append(self.parameters.detach())
        self.momenta.append(self.momentum)
        self.hamiltonians.append(self.hamiltonian_.detach())
        self.rands_.append(uniform_rand)
        self.shadows.append(self.shadow_.detach())
        if self.shadow:
            self.radon_nikodym.append(radon_nikodym.detach())
        return None

    def acceptance_rate(self):
        """Returns the current average acceptance probability amongst generated
        samples"""
        total = float(self.accepted + self.rejected)
        return self.accepted / total
            
    def fetch_samples(self):
        """Return the obtained samples as a single Torch tensor."""
        return torch.cat(self.samples,dim=0).reshape(-1,self.parameters.numel())
    
    def hamiltonian_rejects(self):
        ratios = torch.cat(self.hamiltonian_error,dim=0).reshape(-1,1).detach()
        rands = torch.cat(self.rands_,dim=0).reshape(-1,1)
        nums = rands>=torch.exp(ratios)
        ham_rejects = nums.sum().item()
#        total = float(self.accepted + self.rejected)
        return ham_rejects
    
        
    def shadow_rejects(self):
        assert self.shadow, 'Requires the shadow Hamiltonian to be turned on.'
        ratios = torch.cat(self.shadow_hamiltonian_error,dim=0).reshape(-1,1).detach()
        rands = torch.cat(self.rands_,dim=0).reshape(-1,1)
        nums = rands>=torch.exp(ratios)
        ham_rejects = nums.sum().item()
#        total = float(self.accepted + self.rejected)
        return ham_rejects
    
    def sample_model(self):
        while not self.completed:
            self.degenerate = False
            self.draw_samples()
        return self.fetch_samples()
    
    def draw_samples(self):
        """Run HMC to obtain samples"""
        if self._integrator == 'HMC':       
            self.momentum = torch.distributions.Normal(torch.zeros_like(self.parameters), torch.ones_like(self.parameters)).sample()
        start = time.time()
        if (self._integrator == 'RMHMC'): #torch has trouble differentiating through repeated eigenvalues
            self.jitters = self.jitter * torch.rand(self.parameters.shape[0])
            self.jitters[0] = 0.
            self.jitters[1] = 0.
        self.potential_ = self.get_potential()
        self.hamiltonian_ = self.get_hamiltonian()
        self.momentum = self.resample_momenta(init=True)
        self.momenta.append(self.momentum)
        if self.shadow:
            self.shadow_ = self.get_shadow()
        finished = 0
        counter = 0
        if self.verbose:
            for sample in range(self.n_samples):
                self.step()
                if self.degenerate:
                    break
                finished += 1
        else:
#            for _ in tqdm(range(self.n_samples)):
            for sample in range(self.n_samples):
                self.step()
                if self.degenerate:
                    break
                finished += 1
                counter += 1
                if counter > self.n_samples * 0.05:
                    counter = 0
                    print('('+str(int((sample+1)/self.n_samples*100))+'% complete)', int(self.accepted),'of', int(self.accepted + self.rejected), 'accepted', '('+str(int((self.accepted)/(self.accepted+self.rejected)*100))+'%)')
        total = float(self.accepted + self.rejected)
        end = time.time()
        if total >= self.n_samples:
            self.completed = True
            self.elapsed += end-start
            print('\n', int(self.accepted), ' of ', int(self.accepted + self.rejected), ' samples accepted in', self.elapsed, ' seconds (', 100 * self.accepted/total,'%).')
            return None
        else:
            self.degenerates +=1
            self.find_mode()
            self.parameters = params_init + torch.randn(self.parameters.shape[0])/100
            self.reinitiate_samples()
            self.resample_momenta(init = True)
            return None
        
    def reinitiate_samples(self):
        self.momenta = []
        self.hamiltonians = []
        self.shadows = []
        self.radon_nikodym = []
        self.hamiltonian_error = []
        self.shadow_hamiltonian_error = []
        self.momentum_accepts_ = []
        self.momentum_accepted_ = []
        self.rands_ = []
        self.samples = [self.parameters.detach()]
        self.rejected = 0
        self.accepted = 0
        
    def plot_pacf(self, dim=0, burn = 50, max_lags = 50):
        plot_pacf(self.fetch_samples()[burn:,dim].detach(), lags=max_lags)
        plt.show()
        
    def ess(self,dim=0,burn=50,clip=0, is_ess=True):
        ess = pymc3.ess(self.fetch_samples()[burn:len(self.samples)-clip,dim].detach().numpy())
        if self.shadow and is_ess == True:
            weights = torch.cat(self.radon_nikodym[burn:len(self.samples)-clip],dim=0).reshape(-1,1).detach()
            normed_weights = weights/weights.sum()
            is_ess = torch.sum(normed_weights**2).reciprocal()
            adjusted_ess = ess*is_ess/weights.shape[0]
            return adjusted_ess
        else:
            return ess
    
    def fetch_weights(self):
        assert self.shadow, 'Importance sampling is only required for shadow Hamiltonian techniques, otherwise all weights are 1.'
        weights = torch.cat(self.radon_nikodym[:],dim=0).reshape(-1,1).detach()
        return weights
    
    def plot_errors(self,burn = 50, probs = False, lower=-2,  upper=2):
        blr_errors = torch.cat(self.hamiltonian_error,dim=0).reshape(-1,1).detach()
        blr_shadow_errors = torch.cat(self.shadow_hamiltonian_error,dim=0).reshape(-1,1).detach()
        if not probs:
            plt.plot(blr_errors[burn:].numpy(),alpha=1,c='#FFDB58')
            plt.plot(blr_shadow_errors[burn:].numpy(),c='#C11B17')
            plt.ylim((lower,upper))
            plt.show()
        else:
            wbcd_experr = torch.exp(blr_errors)
            wbcd_probs = torch.min(wbcd_experr,torch.ones(1))
            wbcd_shadow_experr = torch.exp(blr_shadow_errors)
            wbcd_shadow_probs = torch.min(wbcd_shadow_experr,torch.ones(1))
            plt.plot(wbcd_probs[burn:].numpy(),c='#C58917',alpha=0.5)
            plt.plot(wbcd_shadow_probs[burn:].numpy(),c='#C11B17')
            plt.show()
        
    def find_mode(self, its = 500, step = 0.01):
            self.parameters = self.parameters.detach().requires_grad_(True)
            potential = self.get_potential()
            d_potential = torch.autograd.grad(potential, self.parameters, retain_graph=False)[0]
            self.parameters = self.parameters - step*d_potential
    
    def summarise(self, dim = 0, burn = 50, max_lags = 100):
        print(self.hamiltonian_rejects())
        if self.shadow == True:
            print(self.shadow_rejects())
        print(self.ess(burn=burn))
        if self.shadow == True:    
            print(self.ess(is_ess = False,burn=burn))
        self.plot_pacf(max_lags = max_lags)
        if self.shadow == True:
            self.plot_errors()
            self.plot_errors(probs = True)

    def minESS(self, burn = 50, is_ess = True):
        ess = []
        for d in range(self.parameters.shape[0]):
            ess.append(self.ess(dim = d, burn = burn, is_ess = is_ess))
        return min(ess)
    
    def plot_marginals(self,offset = 0):
        samples = self.fetch_samples()
        plt.figure(figsize=(12,9))
        for i in range(1,16):
            plt.subplot(4,4,i)
            sns.kdeplot(samples[:,i-1-offset].numpy())
        plt.subplot(4,4,16)
        plt.plot(samples[:,:].numpy())
        plt.show

def load_dataset(dataset = 'german', seed = 42):
    if dataset == 'german':
        rawdata = pd.read_csv('german_numeric.csv',sep='\s+',header = None)
        data = rawdata.loc[:,0:23]
        preds = rawdata[24]
        pred = preds - 1
        X_train,X_test, y_train, y_test = train_test_split(data,pred,random_state = seed)
    elif dataset == 'australian':
        rawdata = pd.read_csv('australian.dat',sep='\s+',header = None)
        data = rawdata.loc[:,0:13]
        preds = rawdata[14]
        pred = preds - 1
        X_train,X_test, y_train, y_test = train_test_split(data,preds,random_state = seed)
    elif dataset == 'breast':
        rawdata = pd.read_csv('wdbc.csv',header = None)
        data = rawdata.loc[:,1:]
        preds = rawdata[0]
        X_train,X_test, y_train, y_test = train_test_split(data,preds,random_state = seed)
    elif dataset == 'parkinsons':
        rawdata = pd.read_csv('ReplicatedAcousticFeatures-ParkinsonDatabase.csv', header = None)
        data = rawdata.loc[:,2:]
        preds = rawdata[1]
        X_train,X_test, y_train, y_test = train_test_split(data,preds,random_state = seed)
    elif dataset == 'divorce':
        rawdata = pd.read_csv('divorce.csv', sep = ';', header = None)
        data = rawdata.loc[:,:53]
        preds = rawdata[54]
        X_train,X_test, y_train, y_test = train_test_split(data,preds,random_state = seed)
    elif dataset == 'sonar':
        rawdata = pd.read_csv('sonar.csv',header = None)
        data = rawdata.loc[:,0:59]
        preds = rawdata[60]
        X_train,X_test, y_train, y_test = train_test_split(data,preds,random_state = seed)
    means, stdev = X_train.mean(), X_train.std()
    X_norm = (X_train - means)/stdev
    X_testnorm = (X_test - means)/stdev
    (N,D) = X_norm.shape
    D +=1
    XX = torch.tensor(X_norm.values)
    XT = torch.tensor(X_testnorm.values)
    labels = torch.tensor(y_train.values,dtype = torch.float64)
    bias = torch.ones(N,1)
    XX = torch.cat((bias,XX),1)
    XT = torch.cat((torch.ones(XT.shape[0],1),XT),1)
    return XX,  labels
