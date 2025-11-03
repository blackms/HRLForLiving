"""Logging and monitoring utilities for HRL Finance System"""
import os
from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Logger for tracking training experiments using TensorBoard.
    
    Logs training curves, episode metrics, action distributions, and hyperparameters
    to TensorBoard for visualization and analysis.
    
    Attributes:
        writer: TensorBoard SummaryWriter instance
        log_dir: Directory where logs are saved
        enabled: Whether logging is enabled
    """
    
    def __init__(self, log_dir: str = "runs", experiment_name: Optional[str] = None, enabled: bool = True):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name of the experiment (creates subdirectory)
            enabled: Whether to enable logging (useful for disabling during tests)
        """
        self.enabled = enabled
        
        if not self.enabled:
            self.writer = None
            self.log_dir = None
            return
        
        # Create log directory
        if experiment_name:
            self.log_dir = os.path.join(log_dir, experiment_name)
        else:
            self.log_dir = log_dir
        
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        print(f"TensorBoard logging enabled. Log directory: {self.log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.
        
        Args:
            hparams: Dictionary of hyperparameter names and values
        """
        if not self.enabled or self.writer is None:
            return
        
        # Convert all values to strings for TensorBoard compatibility
        hparams_str = {k: str(v) for k, v in hparams.items()}
        
        # Log as text
        hparams_text = "\n".join([f"{k}: {v}" for k, v in hparams_str.items()])
        self.writer.add_text("Hyperparameters", hparams_text, 0)
    
    def log_episode_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        Log episode-level metrics.
        
        Args:
            episode: Episode number
            metrics: Dictionary of metric names and values
        """
        if not self.enabled or self.writer is None:
            return
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(f"Episode/{metric_name}", float(value), episode)
    
    def log_training_curves(self, episode: int, losses: Dict[str, float]):
        """
        Log training losses and learning curves.
        
        Args:
            episode: Episode number
            losses: Dictionary of loss names and values
        """
        if not self.enabled or self.writer is None:
            return
        
        for loss_name, value in losses.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(f"Training/{loss_name}", float(value), episode)
    
    def log_action_distribution(self, episode: int, actions: np.ndarray, prefix: str = ""):
        """
        Log action distribution statistics.
        
        Args:
            episode: Episode number
            actions: Array of actions (shape: [num_steps, action_dim])
            prefix: Optional prefix for metric names
        """
        if not self.enabled or self.writer is None:
            return
        
        if len(actions) == 0:
            return
        
        actions = np.array(actions)
        
        # Log mean and std for each action dimension
        action_names = ["invest", "save", "consume"]
        for i, name in enumerate(action_names):
            if i < actions.shape[1]:
                mean_val = np.mean(actions[:, i])
                std_val = np.std(actions[:, i])
                
                metric_prefix = f"{prefix}/" if prefix else ""
                self.writer.add_scalar(f"{metric_prefix}Actions/{name}_mean", float(mean_val), episode)
                self.writer.add_scalar(f"{metric_prefix}Actions/{name}_std", float(std_val), episode)
        
        # Log histogram of actions (only if we have enough data)
        if len(actions) > 1:
            for i, name in enumerate(action_names):
                if i < actions.shape[1]:
                    action_values = actions[:, i]
                    # Only log histogram if there's variation in the data
                    if len(action_values) > 0 and (np.max(action_values) - np.min(action_values)) > 1e-8:
                        self.writer.add_histogram(f"Actions/{name}_distribution", action_values, episode)
    
    def log_goal_distribution(self, episode: int, goals: np.ndarray):
        """
        Log goal distribution statistics from high-level agent.
        
        Args:
            episode: Episode number
            goals: Array of goals (shape: [num_steps, goal_dim])
        """
        if not self.enabled or self.writer is None:
            return
        
        if len(goals) == 0:
            return
        
        goals = np.array(goals)
        
        # Log mean and std for each goal dimension
        goal_names = ["target_invest_ratio", "safety_buffer", "aggressiveness"]
        for i, name in enumerate(goal_names):
            if i < goals.shape[1]:
                mean_val = np.mean(goals[:, i])
                std_val = np.std(goals[:, i])
                
                self.writer.add_scalar(f"Goals/{name}_mean", float(mean_val), episode)
                self.writer.add_scalar(f"Goals/{name}_std", float(std_val), episode)
        
        # Log histogram of goals (only if we have enough data)
        if len(goals) > 1:
            for i, name in enumerate(goal_names):
                if i < goals.shape[1]:
                    goal_values = goals[:, i]
                    # Only log histogram if there's variation in the data
                    if len(goal_values) > 0 and (np.max(goal_values) - np.min(goal_values)) > 1e-8:
                        self.writer.add_histogram(f"Goals/{name}_distribution", goal_values, episode)
    
    def log_analytics_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        Log analytics metrics (wealth, stability, etc.).
        
        Args:
            episode: Episode number
            metrics: Dictionary of analytics metric names and values
        """
        if not self.enabled or self.writer is None:
            return
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(f"Analytics/{metric_name}", float(value), episode)
    
    def log_goal_adherence(self, episode: int, goal_adherence: float):
        """
        Log goal adherence metric.
        
        Args:
            episode: Episode number
            goal_adherence: Goal adherence value
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_scalar("Performance/goal_adherence", float(goal_adherence), episode)
    
    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], global_step: int):
        """
        Log multiple scalars at once.
        
        Args:
            tag: Main tag for the scalar group
            scalar_dict: Dictionary of scalar names and values
            global_step: Global step value (e.g., episode number)
        """
        if not self.enabled or self.writer is None:
            return
        
        self.writer.add_scalars(tag, scalar_dict, global_step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
