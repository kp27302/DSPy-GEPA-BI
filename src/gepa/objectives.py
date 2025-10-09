"""Multi-objective evaluation for GEPA optimization."""

from typing import Dict, Any, List, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class Objective:
    """Single optimization objective."""
    name: str
    weight: float
    normalize: bool = True
    inverse: bool = False  # If True, minimize instead of maximize
    metric_fn: Callable[[Dict[str, Any]], float] = None


class MultiObjectiveEvaluator:
    """
    Multi-objective evaluator for GEPA.
    Combines accuracy, test pass rate, cost, and latency into Pareto scoring.
    """
    
    def __init__(self, objectives: List[Objective]):
        """
        Initialize evaluator.
        
        Args:
            objectives: List of Objective instances
        """
        self.objectives = objectives
        self._normalize_cache: Dict[str, tuple] = {}
    
    def evaluate(self, trial_results: Dict[str, Any], 
                 population_results: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate a trial against all objectives.
        
        Args:
            trial_results: Results from a single trial
            population_results: Results from all trials (for normalization)
            
        Returns:
            Dictionary of objective scores
        """
        scores = {}
        
        for obj in self.objectives:
            # Extract raw metric
            if obj.metric_fn:
                raw_value = obj.metric_fn(trial_results)
            else:
                raw_value = trial_results.get(obj.name, 0.0)
            
            # Normalize if requested
            if obj.normalize and population_results:
                raw_value = self._normalize_metric(obj.name, raw_value, population_results)
            
            # Invert if minimizing
            if obj.inverse:
                raw_value = 1.0 - raw_value
            
            scores[obj.name] = raw_value
        
        return scores
    
    def _normalize_metric(self, metric_name: str, value: float,
                         population_results: List[Dict[str, Any]]) -> float:
        """
        Normalize metric value to [0, 1] based on population.
        
        Args:
            metric_name: Name of metric
            value: Raw metric value
            population_results: All population results
            
        Returns:
            Normalized value
        """
        # Extract all values for this metric
        values = []
        for result in population_results:
            if metric_name in result:
                values.append(result[metric_name])
        
        if not values or len(values) < 2:
            return value
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val < 1e-9:
            return 1.0
        
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def compute_weighted_score(self, objective_scores: Dict[str, float]) -> float:
        """
        Compute weighted aggregate score.
        
        Args:
            objective_scores: Dictionary of objective scores
            
        Returns:
            Weighted aggregate score
        """
        total_weight = sum(obj.weight for obj in self.objectives)
        
        weighted_sum = 0.0
        for obj in self.objectives:
            score = objective_scores.get(obj.name, 0.0)
            weighted_sum += score * obj.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def is_pareto_dominated(self, scores_a: Dict[str, float], 
                           scores_b: Dict[str, float]) -> bool:
        """
        Check if scores_a is dominated by scores_b.
        
        Args:
            scores_a: First score vector
            scores_b: Second score vector
            
        Returns:
            True if scores_a is dominated by scores_b
        """
        all_objectives = [obj.name for obj in self.objectives]
        
        # scores_b dominates scores_a if:
        # - scores_b >= scores_a for all objectives
        # - scores_b > scores_a for at least one objective
        
        weakly_dominates = all(
            scores_b.get(obj, 0) >= scores_a.get(obj, 0) 
            for obj in all_objectives
        )
        
        strictly_better_in_one = any(
            scores_b.get(obj, 0) > scores_a.get(obj, 0)
            for obj in all_objectives
        )
        
        return weakly_dominates and strictly_better_in_one
    
    def compute_pareto_frontier(self, 
                               population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute Pareto frontier from population.
        
        Args:
            population: List of evaluated individuals
            
        Returns:
            List of non-dominated individuals
        """
        pareto_front = []
        
        for candidate in population:
            candidate_scores = candidate.get('objective_scores', {})
            
            # Check if candidate is dominated by any member of current front
            is_dominated = False
            for front_member in pareto_front:
                front_scores = front_member.get('objective_scores', {})
                if self.is_pareto_dominated(candidate_scores, front_scores):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove any members of front that are dominated by candidate
                pareto_front = [
                    member for member in pareto_front
                    if not self.is_pareto_dominated(
                        member.get('objective_scores', {}),
                        candidate_scores
                    )
                ]
                pareto_front.append(candidate)
        
        return pareto_front


# Default objectives for BI system
def create_default_objectives() -> List[Objective]:
    """
    Create default objectives for BI optimization.
    
    Returns:
        List of default Objective instances
    """
    return [
        Objective(
            name="accuracy",
            weight=0.6,
            normalize=True,
            inverse=False
        ),
        Objective(
            name="tests",
            weight=0.2,
            normalize=True,
            inverse=False
        ),
        Objective(
            name="cost",
            weight=0.1,
            normalize=True,
            inverse=True  # Minimize cost
        ),
        Objective(
            name="latency",
            weight=0.1,
            normalize=True,
            inverse=True  # Minimize latency
        )
    ]

