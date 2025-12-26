"""
Association Stage

First stage of CAMformer pipeline.
Computes Q*K^T similarity using BA-CAM array with binary attention.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.component import Component
from core.event import Event, EventType
from core.statistics import CounterStatistic, AccumulatorStatistic, AverageStatistic
from components.ba_cam_array import BACAMArray, BACAMConfig
from components.adc import ADC, ADCConfig
from components.buffers import InputBuffer, BufferConfig


@dataclass
class AssociationConfig:
    """Configuration for Association stage"""

    # Sequence parameters (updated to match paper)
    max_seq_length: int = 1024
    head_dim: int = 64

    # BA-CAM configuration
    cam_config: BACAMConfig = field(default_factory=lambda: BACAMConfig(
        num_rows=1024,
        num_cols=64,
    ))

    # ADC configuration
    adc_config: ADCConfig = field(default_factory=lambda: ADCConfig(
        resolution_bits=6,
        num_channels=64,
    ))

    # Buffer configuration
    query_buffer_depth: int = 8

    # Binarization threshold
    binarization_threshold: float = 0.0  # Sign function: > 0 -> +1, else -1


class AssociationStage(Component):
    """
    Association Stage for CAMformer.

    Computes similarity between Query and all Keys using BA-CAM:
    1. Load binarized Keys into BA-CAM array
    2. For each Query, perform parallel CAM search
    3. Output Hamming distances (converted from matchline voltages)

    The BA-CAM enables single-cycle parallel search across all Keys.
    """

    def __init__(self, name: str, config: Optional[AssociationConfig] = None,
                 params: Optional[Dict[str, Any]] = None):
        super().__init__(name, params)

        self.config = config or AssociationConfig()

        # Create subcomponents
        self._ba_cam = BACAMArray(
            f"{name}_cam",
            config=self.config.cam_config
        )

        self._adc = ADC(
            f"{name}_adc",
            config=self.config.adc_config
        )

        self._query_buffer = InputBuffer(
            f"{name}_query_buf",
            config=BufferConfig(
                depth=self.config.query_buffer_depth,
                width=self.config.head_dim,
            )
        )

        # Register subcomponents
        self.set_subcomponent("ba_cam", self._ba_cam)
        self.set_subcomponent("adc", self._adc)
        self.set_subcomponent("query_buffer", self._query_buffer)

        # State
        self._keys_loaded = False
        self._num_keys = 0

        # Setup statistics
        self._setup_statistics()

    def _setup_statistics(self) -> None:
        """Setup statistics"""
        self.add_statistic("queries_processed", CounterStatistic(
            "queries_processed", "Total queries processed", "queries"
        ))
        self.add_statistic("total_cycles", AccumulatorStatistic(
            "total_cycles", "Total stage cycles", "cycles"
        ))
        self.add_statistic("total_energy", AccumulatorStatistic(
            "total_energy", "Total stage energy", "pJ"
        ))

    def _setup_impl(self) -> None:
        """Component setup"""
        self._output.info(f"AssociationStage initialized: "
                         f"max_seq={self.config.max_seq_length}, "
                         f"head_dim={self.config.head_dim}")

    def binarize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Binarize floating-point vectors to {-1, +1}.

        Uses sign function: sign(x - threshold)

        Args:
            vectors: Input vectors

        Returns:
            Binarized vectors in {-1, +1}
        """
        return np.sign(vectors - self.config.binarization_threshold)

    def load_keys(self, keys: np.ndarray) -> int:
        """
        Load Keys into BA-CAM array.

        Keys are binarized before loading.

        Args:
            keys: Key matrix (N, d_k) in floating point

        Returns:
            Latency in cycles
        """
        # Binarize keys
        binary_keys = self.binarize(keys)

        # Load into BA-CAM
        latency = self._ba_cam.write_keys(binary_keys)

        self._keys_loaded = True
        self._num_keys = keys.shape[0]

        self._output.debug(f"Loaded {self._num_keys} keys, latency={latency} cycles")

        return latency

    def process_query(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a single Query through BA-CAM.

        Args:
            query: Query vector (d_k,) in floating point

        Returns:
            Tuple of (hamming_distances, similarity_scores, latency_cycles)
        """
        if not self._keys_loaded:
            raise RuntimeError("Keys not loaded in BA-CAM")

        # Binarize query
        binary_query = self.binarize(query)

        # Perform CAM search
        hamming_distances, matchline_voltages, cam_latency = self._ba_cam.search(
            binary_query
        )

        # Convert matchline voltages to digital via ADC
        digital_codes, adc_latency = self._adc.convert(matchline_voltages)

        # Convert to similarity scores
        max_distance = self.config.head_dim
        similarity_scores = 1.0 - (hamming_distances / max_distance)

        # Total latency
        total_latency = cam_latency + adc_latency

        # Update statistics
        self.get_statistic("queries_processed").add_data(1)
        self.get_statistic("total_cycles").add_data(total_latency)

        # Calculate energy from subcomponents
        cam_stats = self._ba_cam.get_statistics().get_summary()
        adc_stats = self._adc.get_statistics().get_summary()

        return hamming_distances, similarity_scores, total_latency

    def process_all_queries(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process all Queries through BA-CAM.

        Args:
            queries: Query matrix (N, d_k)

        Returns:
            Tuple of (all_hamming_distances, all_similarity_scores, total_latency)
        """
        num_queries = queries.shape[0]

        all_hamming = np.zeros((num_queries, self._num_keys), dtype=np.int32)
        all_similarity = np.zeros((num_queries, self._num_keys), dtype=np.float32)
        total_latency = 0

        for i in range(num_queries):
            hamming, similarity, latency = self.process_query(queries[i])
            all_hamming[i] = hamming
            all_similarity[i] = similarity
            total_latency += latency

        self._output.debug(f"Processed {num_queries} queries, "
                          f"total_latency={total_latency} cycles")

        return all_hamming, all_similarity, total_latency

    def get_area_mm2(self) -> float:
        """Get total stage area"""
        area = (self._ba_cam.get_area_mm2() +
                self._adc.get_area_mm2() +
                self._query_buffer.get_area_mm2())
        return area

    def get_stage_stats(self) -> Dict[str, Any]:
        """Get stage statistics"""
        return {
            'area_mm2': self.get_area_mm2(),
            'keys_loaded': self._num_keys,
            'cam_stats': self._ba_cam.get_hardware_stats(),
            'adc_stats': self._adc.get_hardware_stats(),
        }

    def clear(self) -> None:
        """Clear stage state"""
        self._ba_cam.clear()
        self._query_buffer.clear()
        self._keys_loaded = False
        self._num_keys = 0

    def __str__(self) -> str:
        return (f"AssociationStage({self.name}, "
                f"keys={self._num_keys}, "
                f"head_dim={self.config.head_dim})")
