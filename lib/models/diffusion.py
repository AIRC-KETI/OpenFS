import tqdm
import math
import numpy as np

import torch
import torch.nn as nn

from lib.utils.loss import get_mse_loss_with_ignore
import constants

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def make_ddim_schedule(total_steps, ddim_steps):
    """
    Generate evenly spaced time steps for DDIM sampling.
    Args:
        total_steps: total diffusion steps (e.g., 1000)
        ddim_steps: number of sampling steps (e.g., 50)
    Returns:
        A numpy array of selected time steps
    """
    return np.round(np.linspace(0, total_steps - 1, ddim_steps)).astype(int)


class Diffusion(nn.Module):
    def __init__(
            self, beta_1=1e-4, beta_T=0.02, T=1000,
            simple_loss=get_mse_loss_with_ignore,
            schedule_name="cosine",
            sampling_name='ddim',
            eta=0,
            ddim_steps=50,
        ):
        super().__init__()

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.get_simple_loss = simple_loss
        self.schedule_name = schedule_name
        self.sampling_name = sampling_name
        self.eta = eta
        self.ddim_steps = ddim_steps

        if schedule_name == "linear":
            betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        elif schedule_name == "cosine":
            betas = betas_for_alpha_bar(
                T,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        alphas = 1 - betas
        alpha_bars = torch.cumprod(
            alphas,
            dim = 0
        )

        alpha_prev_bars = torch.cat([torch.Tensor([1]), alpha_bars[:-1]])
        sigmas = torch.sqrt((1 - alpha_prev_bars) / (1 - alpha_bars)) * torch.sqrt(1 - (alpha_bars / alpha_prev_bars))

        posterior_variance = (
            betas * (1.0 - alpha_prev_bars) / (1.0 - alpha_bars)
        )
        posterior_log_variance_clipped = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def forward(
        self, model,
        poses, frame_label, frame_idx,
        timesteps=None, get_target=False, get_losses=False,
    ):
        return_list = []

        B = frame_label.shape[0]
        device = frame_label.device

        if timesteps == None:
            timesteps = torch.randint(0, len(self.alpha_bars), (B, )).to(device)
            used_alpha_bars = self.alpha_bars[timesteps][:, None, None, None]
            epsilon_poses = torch.randn_like(poses)
            x_tilde_poses = torch.sqrt(used_alpha_bars) * poses + torch.sqrt(1 - used_alpha_bars) * epsilon_poses
        else:
            timesteps = torch.Tensor([timesteps for _ in range(B)]).to(device).long()
            x_tilde_poses = poses

        pred_X0_dict \
            = model(
                timesteps, x_tilde_poses, frame_label, frame_idx,
            )
        pred_X0_poses = pred_X0_dict['output']
        return_list.append(pred_X0_poses)
        if get_losses:
            target_poses = poses.clone().detach()
            total_loss = self.get_loss(
                pred_X0_poses, target_poses,
            )
            return_list.append(total_loss)

        if get_target:
            return_list.append(epsilon_poses)
            return_list.append(used_alpha_bars)
        return return_list

    def get_loss(self, pred_X0_poses, tgt_poses):
        valid_mask = tgt_poses != constants.MINUS_TWO_VALUE
        # Loss weight
        simple_loss = self.get_simple_loss(pred_X0_poses, tgt_poses, valid_mask)
        return simple_loss

    @torch.no_grad()
    def sampling(
        self, model,
        frame_label, frame_idx,
        njoints=21, pose_dim=3,
        return_middle=False,
    ):
        sampling_number = len(frame_label)
        max_nframes = len(frame_label[0])
        sample_poses = torch.randn([sampling_number, max_nframes, njoints, pose_dim]).to(frame_label.device)

        if self.sampling_name == 'ddpm':
            sample_poses = self.ddpm_loop(
                model, sample_poses,
                frame_label, frame_idx,
                return_middle
            )
        elif self.sampling_name == 'ddim':
            sample_poses = self.ddim_loop(
                model, sample_poses,
                frame_label, frame_idx,
                return_middle, eta=self.eta,
                ddim_steps=self.ddim_steps,
            )
        return sample_poses

    def ddpm_loop(
            self, model, sample_poses,
            frame_label, frame_idx,
            return_middle
        ):
        for t_idx in tqdm.tqdm(
            reversed(range(len(self.alpha_bars))),
            desc="sampling",
            total=len(self.alpha_bars)
        ):
            noise_poses = torch.zeros_like(sample_poses) if t_idx == 0 else torch.randn_like(sample_poses)

            pred_X0_poses, \
                = self.forward(
                    model,
                    sample_poses, frame_label, frame_idx,
                    timesteps=t_idx,
                )
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_poses = pred_X0_poses*coefficient_X0+sample_poses*coefficient_noise

            sample_poses = mu_xt_poses + torch.exp(0.5*log_variance) * noise_poses
            if return_middle and t_idx == 500:
                return sample_poses
        return sample_poses

    def ddim_loop(
        self, model, sample_poses,
        frame_label, frame_idx,
        return_middle=False,
        eta=0.0,
        ddim_steps=50,
    ):
        device = sample_poses.device
        T = len(self.alpha_bars)  # Total diffusion steps (e.g., 1000)
        t_schedule = make_ddim_schedule(T, ddim_steps)  # Select steps for sampling (e.g., 50 steps)

        for i, t_idx in tqdm.tqdm(
            enumerate(reversed(t_schedule)),
            desc="sampling",
            total=len(t_schedule)
        ):
            # Define previous time step
            t_prev = 0 if i == len(t_schedule) - 1 else t_schedule[-(i+2)]

            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars[t_prev]

            alpha_bar = torch.tensor(alpha_bar, device=device)
            alpha_bar_prev = torch.tensor(alpha_bar_prev, device=device)

            # Predict x₀ from the model
            pred_X0_poses, = self.forward(
                model,
                sample_poses, frame_label, frame_idx,
                timesteps=t_idx,
            )

            # Compute predicted noise (epsilon) from predicted x₀
            eps_hat = (sample_poses - torch.sqrt(alpha_bar) * pred_X0_poses) / torch.sqrt(1 - alpha_bar)

            # Compute sigma_t from eta
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
                            torch.sqrt(1 - alpha_bar / alpha_bar_prev)

            # Generate noise if eta > 0
            noise = torch.randn_like(sample_poses) if eta > 0 and t_idx > 0 else 0.0

            # DDIM update step
            sample_poses = (
                torch.sqrt(alpha_bar_prev) * pred_X0_poses +
                torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_hat +
                sigma_t * noise
            )

            # Optionally return the halfway sample (e.g., at t=500)
            if return_middle and t_idx == 500:
                return sample_poses

        return sample_poses


class DiffusionGlobal(nn.Module):
    def __init__(
            self, beta_1=1e-4, beta_T=0.02, T=1000,
            simple_loss=get_mse_loss_with_ignore,
            schedule_name="cosine",
            sampling_name='ddim',
            eta=0,
            ddim_steps=50,
        ):
        super().__init__()

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.get_simple_loss = simple_loss
        self.schedule_name = schedule_name
        self.sampling_name = sampling_name
        self.eta = eta
        self.ddim_steps = ddim_steps

        if schedule_name == "linear":
            betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
        elif schedule_name == "cosine":
            betas = betas_for_alpha_bar(
                T,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        alphas = 1 - betas
        alpha_bars = torch.cumprod(
            alphas,
            dim = 0
        )

        alpha_prev_bars = torch.cat([torch.Tensor([1]), alpha_bars[:-1]])
        sigmas = torch.sqrt((1 - alpha_prev_bars) / (1 - alpha_bars)) * torch.sqrt(1 - (alpha_bars / alpha_prev_bars))

        posterior_variance = (
            betas * (1.0 - alpha_prev_bars) / (1.0 - alpha_bars)
        )
        posterior_log_variance_clipped = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_prev_bars", alpha_prev_bars)
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def forward(
        self, model,
        poses, word_label, frame_idx,
        timesteps=None, get_target=False, get_losses=False,
    ):
        return_list = []

        B = word_label.shape[0]
        device = word_label.device

        if timesteps == None:
            timesteps = torch.randint(0, len(self.alpha_bars), (B, )).to(device)
            used_alpha_bars = self.alpha_bars[timesteps][:, None, None, None]
            epsilon_poses = torch.randn_like(poses)
            x_tilde_poses = torch.sqrt(used_alpha_bars) * poses + torch.sqrt(1 - used_alpha_bars) * epsilon_poses
        else:
            timesteps = torch.Tensor([timesteps for _ in range(B)]).to(device).long()
            x_tilde_poses = poses

        pred_X0_dict \
            = model(
                timesteps, x_tilde_poses, word_label, poses, frame_idx,
            )
        pred_X0_poses = pred_X0_dict['output']
        return_list.append(pred_X0_poses)
        if get_losses:
            target_poses = poses.clone().detach()
            total_loss = self.get_loss(
                pred_X0_poses, target_poses,
            )
            return_list.append(total_loss)

        if get_target:
            return_list.append(epsilon_poses)
            return_list.append(used_alpha_bars)
        return return_list

    def get_loss(self, pred_X0_poses, tgt_poses):
        valid_mask = tgt_poses != constants.MINUS_TWO_VALUE
        # Loss weight
        simple_loss = self.get_simple_loss(pred_X0_poses, tgt_poses, valid_mask)
        return simple_loss

    @torch.no_grad()
    def sampling(
        self, model,
        word_label, frame_idx,
        njoints=21, pose_dim=3,
        return_middle=False,
    ):
        sampling_number = len(word_label)
        max_nframes = len(frame_idx[0])
        sample_poses = torch.randn([sampling_number, max_nframes, njoints, pose_dim]).to(word_label.device)

        if self.sampling_name == 'ddpm':
            sample_poses = self.ddpm_loop(
                model, sample_poses,
                word_label, frame_idx,
                return_middle
            )
        elif self.sampling_name == 'ddim':
            sample_poses = self.ddim_loop(
                model, sample_poses,
                word_label, frame_idx,
                return_middle, eta=self.eta,
                ddim_steps=self.ddim_steps,
            )
        return sample_poses

    def ddpm_loop(
            self, model, sample_poses,
            frame_label, frame_idx,
            return_middle
        ):
        for t_idx in tqdm.tqdm(
            reversed(range(len(self.alpha_bars))),
            desc="sampling",
            total=len(self.alpha_bars)
        ):
            noise_poses = torch.zeros_like(sample_poses) if t_idx == 0 else torch.randn_like(sample_poses)

            pred_X0_poses, \
                = self.forward(
                    model,
                    sample_poses, frame_label, frame_idx,
                    timesteps=t_idx,
                )
            beta = self.betas[t_idx]
            alpha = self.alphas[t_idx]
            alpha_prev_bar = self.alpha_prev_bars[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            log_variance = self.posterior_log_variance_clipped[t_idx]

            coefficient_X0 = (beta*torch.sqrt(alpha_prev_bar)/(1-alpha_bar))
            coefficient_noise = ((1-alpha_prev_bar)*torch.sqrt(alpha)/(1-alpha_bar))

            mu_xt_poses = pred_X0_poses*coefficient_X0+sample_poses*coefficient_noise

            sample_poses = mu_xt_poses + torch.exp(0.5*log_variance) * noise_poses
            if return_middle and t_idx == 500:
                return sample_poses
        return sample_poses

    def ddim_loop(
        self, model, sample_poses,
        word_label, frame_idx,
        return_middle=False,
        eta=0.0,
        ddim_steps=50,
    ):
        device = sample_poses.device
        T = len(self.alpha_bars)  # Total diffusion steps (e.g., 1000)
        t_schedule = make_ddim_schedule(T, ddim_steps)  # Select steps for sampling (e.g., 50 steps)

        for i, t_idx in tqdm.tqdm(
            enumerate(reversed(t_schedule)),
            desc="sampling",
            total=len(t_schedule)
        ):
            # Define previous time step
            t_prev = 0 if i == len(t_schedule) - 1 else t_schedule[-(i+2)]

            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars[t_prev]

            alpha_bar = torch.tensor(alpha_bar, device=device)
            alpha_bar_prev = torch.tensor(alpha_bar_prev, device=device)

            # Predict x₀ from the model
            pred_X0_poses, = self.forward(
                model,
                sample_poses, word_label, frame_idx,
                timesteps=t_idx,
            )

            # Compute predicted noise (epsilon) from predicted x₀
            eps_hat = (sample_poses - torch.sqrt(alpha_bar) * pred_X0_poses) / torch.sqrt(1 - alpha_bar)

            # Compute sigma_t from eta
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
                            torch.sqrt(1 - alpha_bar / alpha_bar_prev)

            # Generate noise if eta > 0
            noise = torch.randn_like(sample_poses) if eta > 0 and t_idx > 0 else 0.0

            # DDIM update step
            sample_poses = (
                torch.sqrt(alpha_bar_prev) * pred_X0_poses +
                torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_hat +
                sigma_t * noise
            )

            # Optionally return the halfway sample (e.g., at t=500)
            if return_middle and t_idx == 500:
                return sample_poses

        return sample_poses