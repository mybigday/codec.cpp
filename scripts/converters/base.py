"""
Base converter class for audio codec models.

All model-specific converters should inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import numpy as np

try:
    from utils import GGUFWriter
except ImportError:  # pragma: no cover - fallback for package-style imports
    from ..utils import GGUFWriter


class BaseConverter(ABC):
    """
    Abstract base class for audio codec GGUF converters.
    
    Subclasses must implement:
    - model_type property
    - architecture property  
    - load_from_checkpoint()
    - load_from_huggingface()
    - convert_and_save()
    """
    
    def __init__(
        self,
        quantization: str = "F16",
        quantize_codebook: bool = False,
        verbose: bool = False
    ):
        """
        Initialize converter.
        
        Args:
            quantization: Quantization type ("F32", "F16", "Q8_0", "Q4_K_M", "Q5_K_M")
            quantize_codebook: Whether to quantize codebook embeddings (default: False)
            verbose: Enable verbose output
        """
        self.quantization = quantization
        self.quantize_codebook = quantize_codebook
        self.verbose = verbose
        self._quantized_tensors = 0
        
        # To be populated by load methods
        self.state_dict: Optional[Dict[str, np.ndarray]] = None
        self.config: Optional[Dict[str, Any]] = None
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type string (e.g., 'mimi', 'dac', 'wavtokenizer')."""
        pass
    
    @property
    @abstractmethod
    def architecture(self) -> str:
        """Return GGUF architecture name."""
        pass
    
    @abstractmethod
    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        """
        Load model from local checkpoint directory.
        
        Must populate:
        - self.state_dict: Dict[str, np.ndarray]
        - self.config: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def load_from_huggingface(self, model_id: str) -> None:
        """
        Load model from HuggingFace.
        
        Must populate:
        - self.state_dict: Dict[str, np.ndarray]
        - self.config: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def convert_and_save(self, output_path: Path) -> None:
        """
        Convert loaded model to GGUF and save.
        
        Args:
            output_path: Path to output .gguf file
        """
        pass
    
    def log(self, message: str) -> None:
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def should_quantize_tensor(self, name: str, arr: np.ndarray) -> bool:
        """
        Determine if a tensor should be quantized.
        
        Default rules:
        - Quantize: weight tensors (.w), conv/linear weights
        - Skip: bias (.b), layer norm, codebook embeddings
        """
        import re
        
        # Never quantize bias
        if name.endswith(".b"):
            return False
        
        # Never quantize layer norm weights
        if re.search(r"(?:_ln|inln|paln)\.w$", name):
            return False
        
        # Never quantize codebook unless explicitly enabled
        if re.search(r"^q\..*\.embed", name) and not self.quantize_codebook:
            return False
        
        # Never quantize small tensors
        if arr.ndim < 2:
            return False
        
        # Check divisibility for quantization
        ne0 = int(arr.shape[0])
        if self.quantization in ("Q4_K_M", "Q5_K_M"):
            return (ne0 % 256) == 0 and name.endswith(".w")
        elif self.quantization == "Q8_0":
            return (ne0 % 32) == 0 and name.endswith(".w")
        
        return False
    
    def _get_target_dtype(self, name: str, arr: np.ndarray) -> str:
        """Determine target dtype for a tensor."""
        if self.quantization == "F32":
            return "F32"
        
        if self.should_quantize_tensor(name, arr):
            return self.quantization
        
        return "F16"

    def _reset_quant_stats(self) -> None:
        self._quantized_tensors = 0

    def _track_quant(self, name: str, arr: np.ndarray) -> None:
        if self.quantization in ("Q4_K_M", "Q5_K_M", "Q8_0") and self.should_quantize_tensor(name, arr):
            self._quantized_tensors += 1

    def _add_tensor(self, writer: GGUFWriter, name: str, arr: np.ndarray, st_dtype: Optional[str] = None) -> None:
        if st_dtype is None:
            st_dtype = self._get_target_dtype(name, arr)
        self._track_quant(name, arr)
        writer.add_tensor(name, arr, st_dtype)

    def _warn_if_no_quantized(self) -> None:
        if self.quantization in ("Q4_K_M", "Q5_K_M", "Q8_0") and self._quantized_tensors == 0:
            print(
                "Warning: 0 tensors eligible for quantization. This usually means tensor row sizes "
                "do not meet the block-size requirements. Output will remain F16-sized.",
                file=sys.stderr,
            )
