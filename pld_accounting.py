from dp_accounting import pld
#from dp_accounting import privacy_accountant
from dp_accounting.pld import pld_privacy_accountant
from dp_accounting.pld import accountant
from dp_accounting.pld import common
from dp_accounting.rdp import rdp_privacy_accountant
from dp_accounting import dp_event
from scipy import optimize
from prv_accountant import Accountant
import numpy as np

def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float=0.1) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    def compute_epsilon(mu: float) -> float:
        #print(f"Noise multiplier : {mu}")
        #print(f"Sampling P:{sampling_probability}")
        #print(f"Delta: {target_delta}")
        #print(f"Max comps :{num_steps}")
        #print(f"Eps_error:{eps_error/2}")
        #print('\n\n\n')
        acc = Accountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps,
            eps_error=eps_error/2
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Ficanding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        #print("Check:", eps_L)
        eps_L = compute_epsilon(mu_L)[0]
    
    has_converged = False
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error

    return bracket[1]

def compute_noise_multiplier_with_pld(
    *,
    num_examples,
    batch_size,
    epochs,
    delta,
    target_epsilon=16.0,
    max_grad_norm=1.0
):
    """
    Compute noise multiplier using Google's DP-PLD for DP-SGD.

    Args:
        num_examples: Dataset size
        batch_size: Size of each minibatch
        epochs: How many times over the data
        noise_multiplier: Gaussian noise stddev / (clipping norm)
        delta: Target delta for (epsilon, delta)-DP

    Returns:
        noise_multiplier: Final noise multiplier at specified delta
    """
    steps = int((num_examples * epochs) // batch_size)
    sampling_probability = batch_size / num_examples

    print(f"Computing noise multiplier for num_examples={num_examples}, batch_size={batch_size}, epochs={epochs}, delta={delta}, target_epsilon={target_epsilon}")
    print(f"Sampling probability: {sampling_probability}, Steps: {steps}")

    all_eps_std_dev = accountant.get_smallest_subsampled_gaussian_noise(
      privacy_parameters=common.DifferentialPrivacyParameters(
          target_epsilon, delta), num_queries=steps, sensitivity=max_grad_norm, sampling_prob=sampling_probability)
    opacus_nm = find_noise_multiplier(
        sampling_probability=sampling_probability,
        num_steps=steps,
        target_epsilon=target_epsilon,
        target_delta=delta
    )
    print(f"Using Opacus-based accountant, noise_multiplier={opacus_nm*max_grad_norm:.3f} for delta={delta:.1e} for epsilon={target_epsilon:.3f}")
    print(f"Using PLD accountant, stddev={all_eps_std_dev:.3f}, stddev normalized={all_eps_std_dev/max_grad_norm:.3f} for delta={delta:.1e} for epsilon={target_epsilon:.3f}")

    return all_eps_std_dev

def compute_privacy_pld_accountant(
    *,
    num_examples,
    batch_size,
    epochs,
    noise_multiplier,
    delta,
    user_epsilon=16.0,
):
    """
    Compute cumulative (epsilon, delta) using Google's DP-PLD for DP-SGD.

    Args:
        num_examples: Dataset size
        batch_size: Size of each minibatch
        epochs: How many times over the data
        noise_multiplier: Gaussian noise stddev / (clipping norm)
        delta: Target delta for (epsilon, delta)-DP

    Returns:
        epsilon: Final epsilon at specified delta
    """
    steps = int((num_examples * epochs) // batch_size)
    sampling_probability = batch_size / num_examples

    # Compose steps with Poisson subsampling
    pld_acc = pld_privacy_accountant.PLDAccountant()
    for i in range(steps):
        pld_acc.compose(dp_event.PoissonSampledDpEvent(sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)))
        
    # Get epsilon for specified delta
    epsilon = pld_acc.get_epsilon(target_delta=delta)
    print(f"Using PLD accountant, epsilon={epsilon:.3f} for delta={delta:.1e}")

    rdp_acc = rdp_privacy_accountant.RdpAccountant()
    for i in range(steps):
        rdp_acc.compose(dp_event.PoissonSampledDpEvent(sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)))
    
    # Get epsilon for specified delta
    epsilon_rdp = rdp_acc.get_epsilon(target_delta=delta)
    print(f"Using RDP accountant, epsilon={epsilon_rdp:.3f} for delta={delta:.1e}")

    return epsilon

# Example usage:
#from datasets import load_from_disk

#export_dir = "/export/fs06/kramesh3/datasets/"
#dataset = load_from_disk(
#    export_dir + "common-pile-news-filtered",
#)
#num_examples = len(dataset['train'])
#batch_size = 4096
#epochs = 25
#epsilon = 4.0
#noise_multiplier = find_noise_multiplier(
#    sampling_probability=batch_size / num_examples,
#    num_steps=int((num_examples * epochs) // batch_size),
#    target_epsilon=epsilon,
#    target_delta=1e-5
#)
#delta = 1/(len(dataset['train'])**1.1)
#print(f"Using noise multiplier: {noise_multiplier:.3f} for delta={delta:.1e}")
#print(f"Number of examples: {num_examples}")
#print(f"Batch size: {batch_size}")
#print(f"Sampling probability: {batch_size / num_examples:.3f}")
#epsilon = compute_privacy_pld_accountant(
#    num_examples=num_examples,
#    batch_size=batch_size,
#    epochs=epochs,
#    noise_multiplier=noise_multiplier,
#    delta=delta, 
#    user_epsilon=epsilon
#)
#Computing noise multiplier for num_examples=50355, batch_size=4096, epochs=5, delta=1e-05, target_epsilon=8.0
if __name__ == "__main__":
    for epsilon in [4.0, 8.0, 16.0, 64.0]:
        print(compute_noise_multiplier_with_pld(
            num_examples=50355,
            batch_size=4096,
            epochs=5,
            delta=1e-5,
            target_epsilon=epsilon, max_grad_norm=0.05
        ))
        print(compute_noise_multiplier_with_pld(
            num_examples=50355,
            batch_size=4096,
            epochs=5,
            delta=1e-5,
            target_epsilon=epsilon, max_grad_norm=1.0
        ))
#for epsilon in [4.0, 8.0, 16.0, 64.0]:
#    noise_multiplier = compute_noise_multiplier_with_pld(
#        num_examples=num_examples,
#        batch_size=batch_size,
#        epochs=epochs,
#        delta=delta,
#       target_epsilon=epsilon,
#        max_grad_norm=1.0
#    )
#print(f"PLD noise multiplier: {noise_multiplier:.3f} under delta={delta:.1e} for epsilon={epsilon:.3f}")
