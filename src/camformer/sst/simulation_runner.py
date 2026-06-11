"""
CAMformer SST Simulation Runner

Orchestrates event-driven simulation runs.
"""

from typing import Optional, Dict, Any, List
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camformer.core.simulation import Simulation
from camformer.core.clock import TimeConverter, ClockFrequencies
from .camformer_sst import CAMformerSST, CAMformerSSTConfig
from .energy_model import EnergyBreakdown


class SimulationRunner:
    """
    Runner for CAMformer SST simulations.

    Provides:
    - Simulation lifecycle management
    - Result collection
    - Performance metrics
    - Comparison utilities
    """

    def __init__(self, config: Optional[CAMformerSSTConfig] = None,
                 name: str = "CAMformer_SST_Sim"):
        """
        Initialize simulation runner.

        Args:
            config: CAMformer configuration
            name: Simulation name
        """
        self.name = name
        self.config = config or CAMformerSSTConfig()

        # Create simulation engine
        self._sim = Simulation(name)

        # Create CAMformer pipeline
        self._pipeline = CAMformerSST(
            "pipeline",
            config=self.config,
            simulation=self._sim
        )

        # Register pipeline with simulation
        self._sim.add_component("pipeline", self._pipeline)

        # Time converter
        self._time_converter = TimeConverter.from_frequency(
            self.config.clock_freq_hz
        )

        # Results storage
        self._results: List[Dict[str, Any]] = []
        self._run_count = 0

    def setup(self) -> None:
        """Setup simulation"""
        self._pipeline.setup()

    def run_single_head(self, Q: np.ndarray, K: np.ndarray,
                        V: np.ndarray) -> Dict[str, Any]:
        """
        Run simulation for single attention head.

        Args:
            Q: Query matrix (N, d_k)
            K: Key matrix (N, d_k)
            V: Value matrix (N, d_v)

        Returns:
            Results dictionary
        """
        self._run_count += 1
        start_time = time.perf_counter()

        # Reset statistics and energy
        self._pipeline.reset_statistics()
        self._pipeline.reset_energy()

        # Run forward pass
        output, stats = self._pipeline.forward_sync(Q, K, V)

        elapsed_time = time.perf_counter() - start_time

        # Get energy metrics
        energy_breakdown = self._pipeline.get_total_energy_breakdown()
        num_queries = Q.shape[0]

        # Collect results
        result = {
            'run_id': self._run_count,
            'output': output,
            'output_shape': output.shape if output is not None else None,
            'latency_cycles': self._pipeline.get_total_latency_cycles(),
            'latency_ns': self._pipeline.get_total_latency_ns(),
            'wall_time_s': elapsed_time,
            'stage_stats': stats,
            'energy': {
                'total_pj': energy_breakdown.total_energy_pj,
                'total_nj': energy_breakdown.total_energy_nj,
                'per_query_pj': energy_breakdown.total_energy_pj / num_queries,
                'queries_per_mj': self._pipeline.get_queries_per_mj(num_queries),
                'breakdown': energy_breakdown.get_breakdown_dict(),
            },
            'config': {
                'seq_length': Q.shape[0],
                'head_dim': Q.shape[1],
                'k_value': self.config.k_value,
            }
        }

        self._results.append(result)
        return result

    def run_multi_head(self, Q: np.ndarray, K: np.ndarray,
                       V: np.ndarray) -> Dict[str, Any]:
        """
        Run simulation for multi-head attention.

        Args:
            Q: Query tensor (num_heads, N, d_k)
            K: Key tensor (num_heads, N, d_k)
            V: Value tensor (num_heads, N, d_v)

        Returns:
            Results dictionary
        """
        self._run_count += 1
        start_time = time.perf_counter()

        num_heads = Q.shape[0]
        outputs = []
        total_cycles = 0

        for h in range(num_heads):
            self._pipeline.reset_statistics()
            self._pipeline.clear()

            output, stats = self._pipeline.forward_sync(Q[h], K[h], V[h])
            outputs.append(output)
            total_cycles += self._pipeline.get_total_latency_cycles()

        elapsed_time = time.perf_counter() - start_time

        outputs = np.stack(outputs, axis=0)

        result = {
            'run_id': self._run_count,
            'output': outputs,
            'output_shape': outputs.shape,
            'num_heads': num_heads,
            'latency_cycles': total_cycles,
            'latency_ns': self._time_converter.cycles_to_ns(total_cycles),
            'wall_time_s': elapsed_time,
            'config': {
                'seq_length': Q.shape[1],
                'head_dim': Q.shape[2],
                'num_heads': num_heads,
                'k_value': self.config.k_value,
            }
        }

        self._results.append(result)
        return result

    def run_sweep(self, seq_lengths: List[int],
                  head_dim: int = 64,
                  k_value: int = 32) -> List[Dict[str, Any]]:
        """
        Run simulation sweep over sequence lengths.

        Args:
            seq_lengths: List of sequence lengths to test
            head_dim: Head dimension
            k_value: Top-k value

        Returns:
            List of results
        """
        sweep_results = []

        for N in seq_lengths:
            # Create new config for this sequence length
            config = CAMformerSSTConfig(
                seq_length=N,
                head_dim=head_dim,
                num_heads=1,
                k_value=min(k_value, N),
            )

            # Create new pipeline
            self._pipeline = CAMformerSST(
                f"pipeline_N{N}",
                config=config,
                simulation=self._sim
            )
            self._pipeline.setup()

            # Generate test data
            Q = np.random.randn(N, head_dim).astype(np.float32)
            K = np.random.randn(N, head_dim).astype(np.float32)
            V = np.random.randn(N, head_dim).astype(np.float32)

            # Run
            result = self.run_single_head(Q, K, V)
            result['sweep_params'] = {'N': N, 'd': head_dim, 'k': k_value}
            sweep_results.append(result)

        return sweep_results

    def compare_with_dense(self, N: int, d: int) -> Dict[str, Any]:
        """
        Compare CAMformer with dense attention.

        Args:
            N: Sequence length
            d: Head dimension

        Returns:
            Comparison dictionary
        """
        k = self.config.k_value

        # Dense attention complexity
        dense_compute = N * N * d  # Q*K^T + Softmax + A*V
        dense_memory = 2 * N * d + N * N + N * d  # Q, K, attention matrix, output

        # CAMformer complexity
        cam_search = N * d  # Binary CAM search (O(1) per query, N queries)
        topk_ops = N * N  # Top-k selection
        sparse_mac = N * k * d  # Sparse A*V

        camformer_compute = cam_search + topk_ops + sparse_mac
        camformer_memory = 2 * N * d + N * k + N * d  # Q, K/CAM, sparse indices, output

        return {
            'dense': {
                'compute_ops': dense_compute,
                'memory_bytes': dense_memory * 2,  # BF16
            },
            'camformer': {
                'compute_ops': camformer_compute,
                'memory_bytes': camformer_memory * 2,
            },
            'speedup': {
                'compute': dense_compute / camformer_compute,
                'memory': dense_memory / camformer_memory,
            },
            'params': {'N': N, 'd': d, 'k': k},
        }

    def get_throughput_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate throughput metrics from result.

        Args:
            result: Simulation result

        Returns:
            Throughput metrics
        """
        latency_ns = result['latency_ns']
        latency_s = latency_ns * 1e-9
        N = result['config']['seq_length']

        return {
            'latency_us': latency_ns / 1000,
            'latency_ms': latency_ns / 1e6,
            'queries_per_second': N / latency_s if latency_s > 0 else 0,
            'tokens_per_second': N / latency_s if latency_s > 0 else 0,
        }

    def print_result(self, result: Dict[str, Any]) -> None:
        """Print formatted result"""
        print(f"\n{'='*60}")
        print(f"Simulation Run #{result['run_id']}")
        print(f"{'='*60}")
        print(f"  Sequence length: {result['config']['seq_length']}")
        print(f"  Head dimension: {result['config']['head_dim']}")
        print(f"  Top-k: {result['config']['k_value']}")
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Latency: {result['latency_cycles']:,} cycles")
        print(f"  Latency: {result['latency_ns']/1000:.2f} us")
        print(f"  Wall time: {result['wall_time_s']*1000:.2f} ms")

        metrics = self.get_throughput_metrics(result)
        print(f"  Throughput: {metrics['queries_per_second']:,.0f} queries/sec")

        # Energy metrics
        if 'energy' in result:
            energy = result['energy']
            print(f"\n  Energy:")
            print(f"    Total: {energy['total_nj']:.2f} nJ ({energy['total_pj']:.0f} pJ)")
            print(f"    Per query: {energy['per_query_pj']:.2f} pJ")
            print(f"    Efficiency: {energy['queries_per_mj']:.0f} qry/mJ")
            if energy.get('breakdown'):
                print(f"    Breakdown:")
                for component, value in energy['breakdown'].items():
                    if component != 'total' and value > 0:
                        print(f"      {component}: {value:.2f} pJ")

    def print_sweep_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print sweep summary table"""
        print(f"\n{'='*90}")
        print("Sequence Length Sweep Summary")
        print(f"{'='*90}")
        print(f"{'N':<8} {'Cycles':<12} {'Latency (us)':<14} {'Energy (nJ)':<12} {'qry/mJ':<12} {'Queries/s':<12}")
        print(f"{'-'*90}")

        for r in results:
            N = r['config']['seq_length']
            cycles = r['latency_cycles']
            latency_us = r['latency_ns'] / 1000
            metrics = self.get_throughput_metrics(r)
            qps = metrics['queries_per_second']

            # Energy metrics
            energy_nj = r.get('energy', {}).get('total_nj', 0)
            qry_per_mj = r.get('energy', {}).get('queries_per_mj', 0)

            print(f"{N:<8} {cycles:<12,} {latency_us:<14.2f} {energy_nj:<12.2f} {qry_per_mj:<12,.0f} {qps:<12,.0f}")

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all simulation results"""
        return self._results.copy()

    def get_pipeline(self) -> CAMformerSST:
        """Get pipeline instance"""
        return self._pipeline

    def get_simulation(self) -> Simulation:
        """Get simulation engine"""
        return self._sim

    def __str__(self) -> str:
        return f"SimulationRunner({self.name}, runs={self._run_count})"
