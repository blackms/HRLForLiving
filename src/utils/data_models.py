"""Data models for HRL system"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    """Represents a single experience tuple for learning"""
    state: np.ndarray          # Current state
    goal: np.ndarray           # Strategic goal (for low-level)
    action: np.ndarray         # Action taken
    reward: float              # Reward received
    next_state: np.ndarray     # Resulting state
    done: bool                 # Episode termination flag
