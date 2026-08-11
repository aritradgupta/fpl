"""
GPU Acceleration & CUDA Device Helper for FPL Optimizer.

Detects NVIDIA CUDA hardware (e.g., NVIDIA GeForce RTX 4060) and provides
vectorized GPU tensor operations for Monte Carlo simulations and parallel Genetic population evaluations.
"""

from typing import Any

import numpy as np

try:
    import torch  # type: ignore[import-untyped,import-not-found]

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_device_info() -> dict[str, Any]:
    """Detect CUDA GPU hardware availability."""
    if HAS_TORCH and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return {
            "gpu_available": True,
            "device": "cuda",
            "name": device_name,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "gpu_available": False,
        "device": "cpu",
        "name": "CPU (PyTorch CUDA not active)",
        "cuda_version": None,
        "device_count": 0,
    }


def simulate_scenarios_gpu(
    base_xp: np.ndarray,
    expected_mins: np.ndarray,
    num_scenarios: int = 1000,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accelerate Monte Carlo match scenario simulation using NVIDIA CUDA GPU tensors.

    Parameters:
        base_xp: 1D array of player baseline xP
        expected_mins: 1D array of player expected minutes
        num_scenarios: Number of Monte Carlo scenario iterations (default 1000)
        use_gpu: Enable CUDA acceleration if available

    Returns:
        (mean_xp_array, variance_xp_array)
    """
    device_info = get_device_info()
    if use_gpu and device_info["gpu_available"]:
        device = torch.device("cuda")
        xp_tensor = torch.tensor(base_xp, dtype=torch.float32, device=device).unsqueeze(1)
        mins_tensor = torch.tensor(expected_mins, dtype=torch.float32, device=device).unsqueeze(1)

        # Generate normal minutes noise & gamma performance noise on GPU CUDA
        num_players = len(base_xp)
        mins_noise = torch.randn((num_players, num_scenarios), device=device) * 15.0 + mins_tensor
        mins_clipped = torch.clamp(mins_noise, 0.0, 90.0)

        # Gamma noise approximation on GPU
        perf_noise = torch.randn((num_players, num_scenarios), device=device).abs() * 0.5 + 0.5
        scenarios = (mins_clipped / 90.0) * xp_tensor * perf_noise

        mean_xp = torch.mean(scenarios, dim=1).cpu().numpy()
        var_xp = torch.var(scenarios, dim=1).cpu().numpy()
        return mean_xp, var_xp

    # CPU Fallback
    rng = np.random.default_rng(seed=42)
    num_players = len(base_xp)
    scenarios_cpu = np.zeros((num_players, num_scenarios))

    for i in range(num_players):
        mins_sim = rng.normal(loc=expected_mins[i], scale=15.0, size=num_scenarios)
        mins_sim = np.clip(mins_sim, 0.0, 90.0)
        perf_noise = rng.gamma(shape=2.0, scale=0.5, size=num_scenarios)
        scenarios_cpu[i] = (mins_sim / 90.0) * base_xp[i] * perf_noise

    mean_xp = np.mean(scenarios_cpu, axis=1)
    var_xp = np.var(scenarios_cpu, axis=1)
    return mean_xp, var_xp
