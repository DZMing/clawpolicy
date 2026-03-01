#!/usr/bin/env python3
"""
Performance optimizer - Batch inference and model quantification

Implement inference performance optimization：
- Batch inference（Batch Inference）
- Model quantification（Quantization）
- caching mechanism（Memoization）
- JITcompile（Optional）

Phase 3.4 - expected150OK
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Callable
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# try to importJITcompile
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = None
    njit = None


class BatchInference:
    """
    batch reasoner

    Combine multiple inference requests into batches，Improve throughput
    """

    def __init__(self,
                 model: Any,
                 batch_size: int = 32,
                 timeout_ms: int = 10):
        """
        Initialize batch reasoner

        Args:
            model: Model（policy_netorvalue_net）
            batch_size: batch size
            timeout_ms: batch timeout（millisecond）
        """
        self.model = model
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

        self.pending_requests: List[Dict[str, Any]] = []
        self.request_counter = 0
        self.last_batch_results: List[Tuple[int, float]] = []

    def predict(self, state: np.ndarray, sync: bool = True) -> float:
        """
        predict（Support batch）

        Args:
            state: input status
            sync: Synchronize or not（No waiting for bulk）

        Returns:
            Prediction results
        """
        if sync:
            # synchronous mode：direct inference
            return self._infer_single(state)

        # asynchronous mode：Join bulk queue
        request_id = self._add_request(state)

        # If the queue is full，Perform batch inference
        if len(self.pending_requests) >= self.batch_size:
            self.last_batch_results = self._flush_batch()

        return request_id

    def _add_request(self, state: np.ndarray) -> int:
        """Add request to queue"""
        request_id = self.request_counter
        self.request_counter += 1

        request = {
            "id": request_id,
            "state": state,
            "result": None
        }

        self.pending_requests.append(request)

        return request_id

    def _infer_single(self, state: np.ndarray) -> float:
        """single inference"""
        if hasattr(self.model, 'forward'):
            return self.model.forward(state)
        elif hasattr(self.model, 'get_action_probs'):
            probs = self.model.get_action_probs(state)
            return float(np.argmax(probs))
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")

    def _infer_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch inference"""
        results = []

        for state in states:
            result = self._infer_single(state)
            results.append(result)

        return np.array(results)

    def _flush_batch(self) -> List[Tuple[int, float]]:
        """Perform batch inference"""
        if not self.pending_requests:
            return []

        # Extract all status
        states = np.array([req["state"] for req in self.pending_requests])

        # Batch inference
        results = self._infer_batch(states)

        # Update request results
        for i, req in enumerate(self.pending_requests):
            req["result"] = results[i]

        # Return results
        output = [(req["id"], req["result"]) for req in self.pending_requests]

        # Clear the queue
        self.pending_requests = []

        return output

    def flush(self) -> List[Tuple[int, float]]:
        """Manually refresh the queue"""
        return self._flush_batch()

    def get_queue_size(self) -> int:
        """Get queue size"""
        return len(self.pending_requests)


class ModelQuantization:
    """
    Model quantification

    Change the model weights fromfloat32Convert toint8，Reduce model size and increase inference speed
    """

    @staticmethod
    def quantize_weights(weights: np.ndarray,
                        bits: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Quantitative weight

        Args:
            weights: original weight（float32）
            bits: Number of quantization bits（8or16）

        Returns:
            (Quantified weight, Quantization parameter)
        """
        if bits == 8:
            # INT8Quantify
            qmin, qmax = -128, 127
            dtype: Any = np.int8
        elif bits == 16:
            # INT16Quantify
            qmin, qmax = -32768, 32767
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported number of quantization bits: {bits}")

        # Calculate scaling factors and zero point
        wmin, wmax = weights.min(), weights.max()
        scale = (wmax - wmin) / (qmax - qmin)
        zero_point = qmin - wmin / scale

        # Quantify
        quantized = np.clip(
            np.round(weights / scale + zero_point),
            qmin, qmax
        ).astype(dtype)

        # Quantization parameter
        params = {
            "scale": float(scale),
            "zero_point": int(zero_point),
            "qmin": qmin,
            "qmax": qmax,
            "bits": bits
        }

        return quantized, params

    @staticmethod
    def dequantize_weights(quantized: np.ndarray,
                          params: Dict[str, Any]) -> np.ndarray:
        """
        Inverse quantization weight

        Args:
            quantized: Quantified weight
            params: Quantization parameter

        Returns:
            The weight after inverse quantization（float32）
        """
        scale = params["scale"]
        zero_point = params["zero_point"]

        return (quantized.astype(np.float32) - zero_point) * scale

    def quantize_model(self, model: Any) -> Dict[str, Any]:
        """
        Quantify the entire model

        Args:
            model: Model（Requires a weight matrix）

        Returns:
            Quantified models and parameters
        """
        quantized_model = {}
        quantization_params = {}

        # Extract model weights（Assume there isget_weightsmethod）
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()

            for layer_name, weight_matrix in weights.items():
                q_weight, q_params = self.quantize_weights(weight_matrix, bits=8)

                quantized_model[layer_name] = q_weight
                quantization_params[layer_name] = q_params
        else:
            logger.warning("⚠️ The model does not supportget_weightsmethod，cannot be quantified")
            return {"original_model": model}

        return {
            "quantized_model": quantized_model,
            "quantization_params": quantization_params,
            "original_model": model
        }

    def estimate_size_reduction(self,
                               original_model: Any,
                               quantized_model: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimated model size reduction

        Args:
            original_model: original model
            quantized_model: quantized model

        Returns:
            size comparison statistics
        """
        # Calculate original size（hypothesisfloat32）
        original_size = 0
        if hasattr(original_model, 'get_weights'):
            weights = original_model.get_weights()
            for weight_matrix in weights.values():
                original_size += weight_matrix.nbytes

        # Calculate the size after quantization（int8）
        quantized_size = 0
        for q_weight in quantized_model["quantized_model"].values():
            quantized_size += q_weight.nbytes

        reduction_ratio = quantized_size / original_size
        size_saved = original_size - quantized_size

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "size_saved_mb": size_saved / (1024 * 1024),
            "reduction_ratio": reduction_ratio,
            "compression_rate": (1 - reduction_ratio) * 100
        }


class InferenceCache:
    """
    inference cache

    Caching results for common states，Avoid double counting
    """

    def __init__(self,
                 model: Any,
                 cache_size: int = 10000):
        """
        Initialize cache

        Args:
            model: Model
            cache_size: cache size
        """
        self.model = model
        self.cache_size = cache_size

        # useLRUcache
        self._cached_predict = lru_cache(maxsize=cache_size)(self._predict_uncached)

        # cache statistics（useLRUCached internal statistics）
        self.total_requests = 0

    def _predict_uncached(self, state_bytes: bytes) -> float:
        """Uncached predictions（internal method）"""
        # frombytesrestore state（simplify：defaultfloat32one-dimensional vector）
        state = np.frombuffer(state_bytes, dtype=np.float32)

        if hasattr(self.model, 'forward'):
            return self.model.forward(state)
        elif hasattr(self.model, 'get_action_probs'):
            probs = self.model.get_action_probs(state)
            return float(np.argmax(probs))
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")

    def predict(self, state: np.ndarray) -> float:
        """
        predict（With cache）

        Args:
            state: input status

        Returns:
            Prediction results
        """
        self.total_requests += 1

        # Calculation statushash
        state_bytes = np.asarray(state, dtype=np.float32).tobytes()

        # useLRUcache
        return self._cached_predict(state_bytes)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_info = self._cached_predict.cache_info()

        total = cache_info.hits + cache_info.misses
        hit_rate = cache_info.hits / total if total > 0 else 0

        return {
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "hit_rate": hit_rate,
            "total_requests": self.total_requests,
            "cache_size": cache_info.currsize
        }

    def clear_cache(self):
        """Clear cache"""
        self._cached_predict.cache_clear()
        self.total_requests = 0


class JITOptimizer:
    """
    JITcompile optimizer

    useNumba JITCompilation speeds up numerical calculations
    """

    def __init__(self):
        """initializationJIToptimizer"""
        self.jit_enabled = NUMBA_AVAILABLE

        if self.jit_enabled:
            logger.info("✅ Numba JITEnabled")
        else:
            logger.warning("⚠️ NumbaNot available，JITOptimization disabled")

    def optimize_function(self, func: Callable) -> Callable:
        """
        Optimization function

        Args:
            func: function to optimize

        Returns:
            Optimized function
        """
        if not self.jit_enabled:
            return func

        # tryJITcompile
        try:
            optimized = njit(func)
            return optimized
        except Exception as e:
            logger.warning(f"⚠️ JITCompilation failed: {e}")
            return func

    def benchmark(self,
                 func: Callable,
                 *args,
                 n_iterations: int = 100) -> Dict[str, float]:
        """
        Performance benchmarks

        Args:
            func: function to test
            *args: Function parameters
            n_iterations: Number of iterations

        Returns:
            Performance statistics
        """
        import time

        # preheat
        for _ in range(10):
            func(*args)

        # test
        start_time = time.time()
        for _ in range(n_iterations):
            func(*args)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / n_iterations

        return {
            "total_time_s": total_time,
            "avg_time_ms": avg_time * 1000,
            "throughput": n_iterations / total_time
        }


class PerformanceOptimizer:
    """
    Performance optimizer（Integrate all optimization technologies）

    Complete performance optimization process：
    1. Batch inference
    2. Model quantification
    3. inference cache
    4. JITcompile
    """

    def __init__(self,
                 model: Any,
                 enable_batch: bool = True,
                 enable_quantization: bool = False,
                 enable_cache: bool = True,
                 enable_jit: bool = False):
        """
        Initialize the performance optimizer

        Args:
            model: Model
            enable_batch: Enable batch inference
            enable_quantization: Enable quantization
            enable_cache: Enable caching
            enable_jit: enableJIT
        """
        self.model = model
        self.enable_batch = enable_batch
        self.enable_quantization = enable_quantization
        self.enable_cache = enable_cache
        self.enable_jit = enable_jit

        # Initialize optimization components
        if enable_batch:
            self.batch_inference = BatchInference(model, batch_size=32)

        if enable_quantization:
            self.quantization = ModelQuantization()
            quantized = self.quantization.quantize_model(model)
            self.quantized_model: Dict[str, Any] = quantized

        if enable_cache:
            self.cache = InferenceCache(model, cache_size=10000)

        if enable_jit:
            self.jit_optimizer = JITOptimizer()

        # Performance statistics
        self.optimization_stats = {
            "batch_inference": {"calls": 0, "total_batched": 0},
            "cache": self.cache.get_cache_stats() if enable_cache else {},
            "quantization": self.quantization.estimate_size_reduction(
                model, self.quantized_model
            ) if enable_quantization else {}
        }

    def predict(self, state: np.ndarray) -> float:
        """
        Optimized forecasts

        Args:
            state: input status

        Returns:
            Prediction results
        """
        # Batch inference
        if self.enable_batch:
            result = self.batch_inference.predict(state)
            self.optimization_stats["batch_inference"]["calls"] += 1
            return result

        # caching inference
        elif self.enable_cache:
            return self.cache.predict(state)

        # Quantitative reasoning
        elif self.enable_quantization:
            # （simplify：Actually requires complete quantitative reasoning logic）
            if hasattr(self.model, 'forward'):
                return self.model.forward(state)

        # default reasoning
        return self.model.forward(state)

    def flush(self):
        """Flush all queues"""
        if self.enable_batch:
            return self.batch_inference.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            "batch_inference": self.optimization_stats["batch_inference"],
            "cache": self.cache.get_cache_stats() if self.enable_cache else {},
            "quantization": self.optimization_stats["quantization"]
        }

        if self.enable_batch:
            stats["batch_inference"]["queue_size"] = self.batch_inference.get_queue_size()

        return stats

    def save_stats(self, filepath: str):
        """Save statistics"""
        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

        logger.info(f"✅ Performance statistics saved: {filepath}")


def main():
    """Test performance optimizer"""
    # Create a simulation model
    class DummyModel:
        def __init__(self):
            self.weights = {
                "layer1": np.random.randn(128, 64),
                "layer2": np.random.randn(64, 10)
            }

        def forward(self, state):
            return float(np.random.random())

        def get_weights(self):
            return self.weights

    model = DummyModel()

    # Create optimizer
    optimizer = PerformanceOptimizer(
        model,
        enable_batch=True,
        enable_quantization=True,
        enable_cache=True,
        enable_jit=False
    )

    print("✅ Performance optimizer created")

    # Test predictions
    for i in range(100):
        state = np.random.randn(17)
        optimizer.predict(state)

    # Get statistics
    stats = optimizer.get_stats()
    print("\n📊 Performance statistics:")
    print(f"   Batch inference calls: {stats['batch_inference']['calls']}")
    print(f"   Cache hit rate: {stats['cache'].get('hit_rate', 0):.2%}")
    if stats['quantization']:
        print(f"   Model compression rate: {stats['quantization'].get('compression_rate', 0):.1f}%")


if __name__ == "__main__":
    main()
