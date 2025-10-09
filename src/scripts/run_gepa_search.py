"""CLI script to run GEPA optimization."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gepa.loop import GEPAOptimizer, GEPAConfig
from src.gepa.objectives import create_default_objectives, MultiObjectiveEvaluator
from src.gepa.components import PromptGenome
from src.utils.io import PathManager, load_config, save_json
from rich.console import Console

console = Console()


def dummy_eval_function(genome: PromptGenome) -> dict:
    """
    Dummy evaluation function for demonstration.
    In production, this would run DSPy programs on benchmark tasks.
    """
    import random
    import time
    
    # Simulate evaluation
    time.sleep(0.1)
    
    return {
        'accuracy': random.uniform(0.6, 0.95),
        'tests': random.uniform(0.7, 1.0),
        'cost': random.uniform(100, 500),
        'latency': random.uniform(0.5, 2.0)
    }


def main():
    """Run GEPA optimization search."""
    console.print("[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]  BI-DSPy-GEPA: Optimization Search    [/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")
    
    try:
        # Load config
        config = load_config()
        gepa_config_dict = config.get('gepa', {})
        
        # Create GEPA config
        gepa_config = GEPAConfig(
            population_size=gepa_config_dict.get('population', 16),
            generations=gepa_config_dict.get('generations', 8),
            max_trials=gepa_config_dict.get('budget', {}).get('max_trials', 150),
            mutation_rate=gepa_config_dict.get('mutation_rates', {}).get('swap_fewshot', 0.3),
            crossover_rate=0.5,
            seed=42
        )
        
        console.print(f"[bold]Configuration:[/bold]")
        console.print(f"  Population: {gepa_config.population_size}")
        console.print(f"  Generations: {gepa_config.generations}")
        console.print(f"  Max trials: {gepa_config.max_trials}\n")
        
        # Create objectives
        objectives = create_default_objectives()
        evaluator = MultiObjectiveEvaluator(objectives)
        
        # Create optimizer
        optimizer = GEPAOptimizer(
            config=gepa_config,
            evaluator=evaluator,
            eval_function=dummy_eval_function
        )
        
        # Run optimization
        results = optimizer.optimize()
        
        # Save results
        path_mgr = PathManager()
        results_path = path_mgr.eval_results / "gepa_results.json"
        
        # Convert results for JSON serialization (convert numpy types to Python types)
        import numpy as np
        
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_results = {
            'best_genome': results['best_genome'],
            'pareto_archive': [
                {
                    'genome_id': item['genome'].genome_id,
                    'generation': int(item['generation']),
                    'metrics': convert_numpy(item['metrics']),
                    'objective_scores': convert_numpy(item['objective_scores']),
                    'weighted_score': float(item['weighted_score'])
                }
                for item in results['pareto_archive']
            ],
            'history': convert_numpy(results['history']),
            'total_trials': int(results['total_trials']),
            'mined_rules': results['mined_rules']
        }
        
        save_json(json_results, results_path)
        
        console.print(f"\n[bold green]Results saved to:[/bold green] {results_path}")
        console.print(f"\n[bold]Best Genome:[/bold]")
        
        if results['best_genome']:
            console.print(f"  ID: {results['best_genome']['genome_id']}")
            console.print(f"  Generation: {results['best_genome']['generation']}")
            console.print(f"  Components: {len(results['best_genome']['components'])}")
        
        console.print(f"\n[bold]Pareto Archive:[/bold] {len(results['pareto_archive'])} genomes")
        console.print(f"[bold]Mined Rules:[/bold] {len(results['mined_rules'])} rules")
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red][FAIL] GEPA Search Failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

