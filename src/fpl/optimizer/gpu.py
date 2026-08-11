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
    """
    device_info = get_device_info()
    if use_gpu and device_info["gpu_available"]:
        device = torch.device("cuda")
        xp_tensor = torch.tensor(base_xp, dtype=torch.float32, device=device).unsqueeze(1)
        mins_tensor = torch.tensor(expected_mins, dtype=torch.float32, device=device).unsqueeze(1)

        num_players = len(base_xp)
        mins_noise = torch.randn((num_players, num_scenarios), device=device) * 15.0 + mins_tensor
        mins_clipped = torch.clamp(mins_noise, 0.0, 90.0)

        gpu_perf_noise = torch.randn((num_players, num_scenarios), device=device).abs() * 0.5 + 0.5
        scenarios = (mins_clipped / 90.0) * xp_tensor * gpu_perf_noise

        mean_res: np.ndarray = torch.mean(scenarios, dim=1).cpu().numpy()
        var_res: np.ndarray = torch.var(scenarios, dim=1).cpu().numpy()
        return mean_res, var_res

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


def evaluate_population_gpu(
    population_matrix: np.ndarray,
    xp_vector: np.ndarray,
    cost_vector: np.ndarray,
    budget: float = 100.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Vectorized CUDA GPU evaluation of population chromosome fitnesses.

    Parameters:
        population_matrix: 2D array (pop_size, 15) containing player index IDs
        xp_vector: 1D array of expected points indexed by player position
        cost_vector: 1D array of player costs indexed by player position
        budget: Squad budget limit in £m
        use_gpu: Enable CUDA acceleration if available

    Returns:
        1D array of fitness scores (pop_size,)
    """
    device_info = get_device_info()
    pop_size, squad_size = population_matrix.shape

    if use_gpu and device_info["gpu_available"]:
        device = torch.device("cuda")
        xp_t = torch.tensor(xp_vector, dtype=torch.float32, device=device)
        cost_t = torch.tensor(cost_vector, dtype=torch.float32, device=device)
        pop_t = torch.tensor(population_matrix, dtype=torch.long, device=device)

        # Batch index lookup on GPU
        pop_xp = xp_t[pop_t]  # Shape (pop_size, 15)
        pop_cost = cost_t[pop_t]  # Shape (pop_size, 15)

        total_xp = torch.sum(pop_xp, dim=1)
        total_cost = torch.sum(pop_cost, dim=1)

        # Vectorized constraint mask
        valid_mask = total_cost <= budget
        fitness_t = torch.where(valid_mask, total_xp, torch.tensor(0.0, device=device))
        fit_res: np.ndarray = fitness_t.cpu().numpy()
        return fit_res

    # CPU Vectorized Fallback
    fitnesses = np.zeros(pop_size)
    for i in range(pop_size):
        chrom = population_matrix[i]
        cost = cost_vector[chrom].sum()
        if cost <= budget and len(set(chrom)) == squad_size:
            fitnesses[i] = xp_vector[chrom].sum()
    return fitnesses
