import math
import torch
import importlib

# Try to import trange from comfy.utils, fallback to standard tqdm if missing
try:
    from comfy.utils import model_trange as trange
except ImportError:
    from tqdm.auto import trange

import comfy.samplers
import comfy.model_patcher
import comfy.model_sampling

from comfy.k_diffusion import sampling as k_sampling
from comfy.k_diffusion import utils as k_utils

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / k_utils.append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def default_noise_sampler(x, seed=None):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1

        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)."""
    if hasattr(comfy.model_sampling, 'CONST') and isinstance(model_sampling, getattr(comfy.model_sampling, 'CONST')):
        return sigma.logit().neg()
    return sigma.log().neg()

def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma."""
    if hasattr(comfy.model_sampling, 'CONST') and isinstance(model_sampling, getattr(comfy.model_sampling, 'CONST')):
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()

def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR."""
    if len(sigmas) <= 1:
        return sigmas
    if hasattr(comfy.model_sampling, 'CONST') and isinstance(model_sampling, getattr(comfy.model_sampling, 'CONST')):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas

# ==============================================================================
# SA-Solver Implementation
# ==============================================================================

def compute_exponential_coeffs(s: torch.Tensor, t: torch.Tensor, solver_order: int, tau_t: float) -> torch.Tensor:
    tau_mul = 1 + tau_t ** 2
    h = t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)

    product_terms_factored = (t ** p - s ** p * (-tau_mul * h).exp())

    recursive_depth_mat = p.unsqueeze(1) - p.unsqueeze(0)
    log_factorial = (p + 1).lgamma()
    recursive_coeff_mat = log_factorial.unsqueeze(1) - log_factorial.unsqueeze(0)
    if tau_t > 0:
        recursive_coeff_mat = recursive_coeff_mat - (recursive_depth_mat * math.log(tau_mul))
    signs = torch.where(recursive_depth_mat % 2 == 0, 1.0, -1.0)
    recursive_coeff_mat = (recursive_coeff_mat.exp() * signs).tril()

    return recursive_coeff_mat @ product_terms_factored

def compute_simple_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, is_corrector_step: bool = False) -> torch.Tensor:
    tau_mul = 1 + tau_t ** 2
    h = lambda_t - lambda_s
    alpha_t = sigma_next * lambda_t.exp()
    if is_corrector_step:
        b_1 = alpha_t * (0.5 * tau_mul * h)
        b_2 = alpha_t * (-h * tau_mul).expm1().neg() - b_1
    else:
        b_2 = alpha_t * (0.5 * tau_mul * h ** 2) / (curr_lambdas[-2] - lambda_s)
        b_1 = alpha_t * (-h * tau_mul).expm1().neg() - b_2
    return torch.stack([b_2, b_1])

def compute_stochastic_adams_b_coeffs(sigma_next: torch.Tensor, curr_lambdas: torch.Tensor, lambda_s: torch.Tensor, lambda_t: torch.Tensor, tau_t: float, simple_order_2: bool = False, is_corrector_step: bool = False) -> torch.Tensor:
    num_timesteps = curr_lambdas.shape[0]

    if simple_order_2 and num_timesteps == 2:
        return compute_simple_stochastic_adams_b_coeffs(sigma_next, curr_lambdas, lambda_s, lambda_t, tau_t, is_corrector_step)

    exp_integral_coeffs = compute_exponential_coeffs(lambda_s, lambda_t, num_timesteps, tau_t)
    vandermonde_matrix_T = torch.vander(curr_lambdas, num_timesteps, increasing=True).T
    lagrange_integrals = torch.linalg.solve(vandermonde_matrix_T, exp_integral_coeffs)

    alpha_t = sigma_next * lambda_t.exp()
    return alpha_t * lagrange_integrals

def get_tau_interval_func(start_sigma: float, end_sigma: float, eta: float = 1.0):
    def tau_func(sigma) -> float:
        if eta <= 0:
            return 0.0
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()
        return eta if start_sigma >= sigma >= end_sigma else 0.0
    return tau_func


@torch.no_grad()
def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, use_pece=False, simple_order_2=False):
    """Stochastic Adams Solver with predictor-corrector method (NeurIPS 2023)."""
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    if tau_func is None:
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x

    h = 0.0
    tau_t = 0.0
    noise = 0.0
    pred_list = []

    lower_order_to_end = sigmas[-1].item() == 0

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        pred_list.append(denoised)
        pred_list = pred_list[-max_used_order:]

        predictor_order_used = min(predictor_order, len(pred_list))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(pred_list))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = lambdas[i - corrector_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i],
                curr_lambdas,
                lambdas[i - 1],
                lambdas[i],
                tau_t,
                simple_order_2,
                is_corrector_step=True,
            )
            pred_mat = torch.stack(pred_list[-corrector_order_used:], dim=1)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                x = x + noise

            if use_pece:
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                pred_list[-1] = denoised

        if sigmas[i + 1] == 0:
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = lambdas[i - predictor_order_used + 1:i + 1]
            b_coeffs = compute_stochastic_adams_b_coeffs(
                sigmas[i + 1],
                curr_lambdas,
                lambdas[i],
                lambdas[i + 1],
                tau_t,
                simple_order_2,
                is_corrector_step=False,
            )
            pred_mat = torch.stack(pred_list[-predictor_order_used:], dim=1)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x_pred


@torch.no_grad()
def sample_sa_solver_pece(model, x, sigmas, extra_args=None, callback=None, disable=False, tau_func=None, s_noise=1.0, noise_sampler=None, predictor_order=3, corrector_order=4, simple_order_2=False):
    return sample_sa_solver(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, tau_func=tau_func, s_noise=s_noise, noise_sampler=noise_sampler, predictor_order=predictor_order, corrector_order=corrector_order, use_pece=True, simple_order_2=simple_order_2)

# ==============================================================================
# RES Multistep Implementation
# ==============================================================================

@torch.no_grad()
def res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None, eta=1., cfg_pp=False):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    old_sigma_down = None
    old_denoised = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    if cfg_pp:
        if hasattr(comfy.model_patcher, 'set_model_options_post_cfg_function'):
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
        else:
            # Fallback if the user has an extremely old ComfyUI without post_cfg_function support
            cfg_pp = False

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0 or old_denoised is None:
            if cfg_pp and uncond_denoised is not None:
                d = to_d(x, sigmas[i], uncond_denoised)
                x = denoised + d * sigma_down
            else:
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
        else:
            t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - t
            c2 = (t_prev - t_old) / h

            phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

            if cfg_pp and uncond_denoised is not None:
                x = x + (denoised - uncond_denoised)
                x = sigma_fn(h) * x + h * (b1 * uncond_denoised + b2 * old_denoised)
            else:
                x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if cfg_pp and uncond_denoised is not None:
            old_denoised = uncond_denoised
        else:
            old_denoised = denoised
        old_sigma_down = sigma_down
    return x

@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=True)

@torch.no_grad()
def sample_res_multistep_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=False)

@torch.no_grad()
def sample_res_multistep_ancestral_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=eta, cfg_pp=True)

# Register into ComfyUI
def register_extra_samplers():
    added = 0
    from comfy.samplers import KSampler
    
    samplers_to_add = {
        "sa_solver": sample_sa_solver,
        "sa_solver_pece": sample_sa_solver_pece,
        "res_multistep": sample_res_multistep,
        "res_multistep_cfg_pp": sample_res_multistep_cfg_pp,
        "res_multistep_ancestral": sample_res_multistep_ancestral,
        "res_multistep_ancestral_cfg_pp": sample_res_multistep_ancestral_cfg_pp,
    }

    for name, func in samplers_to_add.items():
        if name not in KSampler.SAMPLERS:
            try:
                # Add it to the global KSampler list so that the UI can use it
                idx = KSampler.SAMPLERS.index("uni_pc_bh2")
                KSampler.SAMPLERS.insert(idx+1, name)
            except ValueError:
                KSampler.SAMPLERS.append(name)
            
            # Register it onto the target module if it's absent
            if not hasattr(k_sampling, f"sample_{name}"):
                setattr(k_sampling, f"sample_{name}", func)
                added += 1

    if added > 0:
        importlib.reload(k_sampling)
        print(f"[UmeAiRT-Toolkit] Registered {added} extra samplers (e.g., sa_solver_pece, res_multistep) for older ComfyUI installations.")

