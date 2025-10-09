"""GEPA main optimization loop: Genetic-Pareto search."""

import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import random
import numpy as np
from dataclasses import dataclass
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import copy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gepa.objectives import MultiObjectiveEvaluator, Objective, create_default_objectives
from src.gepa.components import PromptGenome, TextComponent, create_baseline_genome
from src.gepa.reflectors import Reflector, RuleMiner
from src.utils.seed import set_seed
from src.utils.io import save_json, load_json

console = Console()


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimization."""
    population_size: int = 16
    generations: int = 8
    max_trials: int = 150
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    seed: int = 42
    archive_size: int = 10


class GEPAOptimizer:
    """
    GEPA (Genetic-Pareto) optimizer for prompt evolution.
    
    Evolves prompt genomes to maximize multi-objective fitness:
    - Accuracy ↑
    - Test pass rate ↑
    - Cost ↓
    - Latency ↓
    
    Uses Pareto dominance for selection and maintains archive of non-dominated solutions.
    """
    
    def __init__(self, 
                 config: GEPAConfig,
                 evaluator: MultiObjectiveEvaluator,
                 eval_function: Callable[[PromptGenome], Dict[str, Any]]):
        """
        Initialize GEPA optimizer.
        
        Args:
            config: GEPA configuration
            evaluator: Multi-objective evaluator
            eval_function: Function to evaluate a genome (returns metrics dict)
        """
        self.config = config
        self.evaluator = evaluator
        self.eval_function = eval_function
        
        self.population: List[PromptGenome] = []
        self.archive: List[Dict[str, Any]] = []  # Pareto archive
        self.generation = 0
        self.trial_count = 0
        
        self.reflector = Reflector()
        self.rule_miner = RuleMiner()
        
        set_seed(config.seed)
    
    def initialize_population(self) -> List[PromptGenome]:
        """
        Initialize population with baseline and variants.
        
        Returns:
            Initial population of genomes
        """
        console.print("\n[bold blue]>> Initializing Population[/bold blue]\n")
        
        population = []
        
        # Start with baseline
        baseline = create_baseline_genome()
        baseline.genome_id = f"gen0_baseline"
        baseline.generation = 0
        population.append(baseline)
        
        # Create variants via mutation
        for i in range(self.config.population_size - 1):
            variant = self._mutate(baseline)
            variant.genome_id = f"gen0_variant{i}"
            variant.generation = 0
            population.append(variant)
        
        console.print(f"  [OK] Created population of {len(population)} genomes\n")
        
        return population
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run GEPA optimization loop.
        
        Returns:
            Dictionary with results: best genomes, pareto front, history
        """
        console.print("[bold green]>> Starting GEPA Optimization[/bold green]\n")
        
        # Initialize
        self.population = self.initialize_population()
        history = []
        
        # Evolution loop
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            gen_task = progress.add_task(
                "[cyan]Evolving genomes...",
                total=self.config.generations
            )
            
            for gen in range(self.config.generations):
                self.generation = gen
                
                console.print(f"\n[bold]Generation {gen + 1}/{self.config.generations}[/bold]\n")
                
                # Evaluate population
                gen_results = self._evaluate_population(self.population)
                
                # Update archive with non-dominated solutions
                self._update_archive(gen_results)
                
                # Reflect and mine rules
                self._reflect_on_generation(gen_results)
                
                # Record generation stats
                gen_stats = self._compute_generation_stats(gen_results)
                history.append(gen_stats)
                
                if gen_stats and 'best_weighted_score' in gen_stats:
                    console.print(f"  Best weighted score: {gen_stats['best_weighted_score']:.3f}")
                else:
                    console.print(f"  [yellow][WARN] No successful evaluations this generation[/yellow]")
                console.print(f"  Archive size: {len(self.archive)}")
                console.print(f"  Trials so far: {self.trial_count}\n")
                
                # Check budget
                if self.trial_count >= self.config.max_trials:
                    console.print("[yellow][WARN] Trial budget exhausted[/yellow]")
                    break
                
                # Selection and variation
                if gen < self.config.generations - 1:
                    self.population = self._evolve_population(gen_results)
                
                progress.update(gen_task, advance=1)
        
        # Final results
        best_genome = self._get_best_genome()
        
        results = {
            'best_genome': best_genome.to_dict() if best_genome else None,
            'pareto_archive': self.archive[:self.config.archive_size],
            'history': history,
            'total_trials': self.trial_count,
            'mined_rules': [r.to_dict() for r in self.reflector.rules]
        }
        
        console.print("\n[bold green][SUCCESS] Optimization Complete[/bold green]\n")
        
        return results
    
    def _evaluate_population(self, population: List[PromptGenome]) -> List[Dict[str, Any]]:
        """Evaluate all genomes in population."""
        results = []
        
        for genome in population:
            try:
                # Evaluate genome
                start_time = time.time()
                eval_result = self.eval_function(genome)
                latency = time.time() - start_time
                
                # Add latency to results
                eval_result['latency'] = latency
                
                # Compute objective scores
                obj_scores = self.evaluator.evaluate(eval_result, results)
                weighted_score = self.evaluator.compute_weighted_score(obj_scores)
                
                result = {
                    'genome': genome,
                    'metrics': eval_result,
                    'objective_scores': obj_scores,
                    'weighted_score': weighted_score,
                    'generation': self.generation
                }
                
                results.append(result)
                self.trial_count += 1
                
                # Add to reflector
                self.reflector.add_trace({
                    'genome_id': genome.genome_id,
                    'passed': eval_result.get('accuracy', 0) > 0.7,
                    'metrics': eval_result,
                    'objective_scores': obj_scores
                })
                
            except Exception as e:
                console.print(f"  [red]Error evaluating {genome.genome_id}: {e}[/red]")
                continue
        
        return results
    
    def _update_archive(self, gen_results: List[Dict[str, Any]]):
        """Update Pareto archive with non-dominated solutions."""
        # Combine with existing archive
        all_candidates = self.archive + gen_results
        
        # Compute Pareto frontier
        pareto_front = self.evaluator.compute_pareto_frontier(all_candidates)
        
        # Update archive (keep top N by weighted score)
        pareto_front.sort(key=lambda x: x['weighted_score'], reverse=True)
        self.archive = pareto_front[:self.config.archive_size * 2]
    
    def _reflect_on_generation(self, gen_results: List[Dict[str, Any]]):
        """Reflect on generation results and mine rules."""
        # Mine rules every few generations
        if self.generation % 2 == 0:
            self.reflector.reflect()
    
    def _compute_generation_stats(self, gen_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for a generation."""
        if not gen_results:
            return {}
        
        weighted_scores = [r['weighted_score'] for r in gen_results]
        
        stats = {
            'generation': self.generation,
            'population_size': len(gen_results),
            'best_weighted_score': max(weighted_scores),
            'mean_weighted_score': np.mean(weighted_scores),
            'std_weighted_score': np.std(weighted_scores)
        }
        
        # Add objective-specific stats
        for obj in self.evaluator.objectives:
            obj_scores = [r['objective_scores'].get(obj.name, 0) for r in gen_results]
            stats[f'best_{obj.name}'] = max(obj_scores)
            stats[f'mean_{obj.name}'] = np.mean(obj_scores)
        
        return stats
    
    def _evolve_population(self, gen_results: List[Dict[str, Any]]) -> List[PromptGenome]:
        """Create next generation via selection and variation."""
        new_population = []
        
        # Handle case where no successful evaluations
        if not gen_results:
            console.print("[yellow][WARN] No successful evaluations - keeping current population[/yellow]")
            return self.population[:self.config.population_size]
        
        # Elitism: keep best genomes
        gen_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        elite_count = max(1, min(len(gen_results), self.config.population_size // 4))
        
        for i in range(elite_count):
            elite = copy.deepcopy(gen_results[i]['genome'])
            elite.genome_id = f"gen{self.generation + 1}_elite{i}"
            elite.generation = self.generation + 1
            new_population.append(elite)
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(gen_results)
            parent2 = self._tournament_select(gen_results)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                offspring = self._mutate(offspring)
            
            offspring.genome_id = f"gen{self.generation + 1}_ind{len(new_population)}"
            offspring.generation = self.generation + 1
            offspring.parent_ids = [parent1.genome_id, parent2.genome_id]
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_select(self, gen_results: List[Dict[str, Any]], k: int = 3) -> PromptGenome:
        """Tournament selection."""
        tournament = random.sample(gen_results, min(k, len(gen_results)))
        winner = max(tournament, key=lambda x: x['weighted_score'])
        return winner['genome']
    
    def _crossover(self, parent1: PromptGenome, parent2: PromptGenome) -> PromptGenome:
        """Crossover: combine components from two parents."""
        offspring_components = []
        
        # For each component type, randomly choose from parent1 or parent2
        all_types = set(c.component_type for c in parent1.components + parent2.components)
        
        for comp_type in all_types:
            p1_comps = parent1.get_components_by_type(comp_type)
            p2_comps = parent2.get_components_by_type(comp_type)
            
            if p1_comps and p2_comps:
                # Mix: take some from each
                selected = random.sample(p1_comps, len(p1_comps) // 2 + 1) + \
                          random.sample(p2_comps, len(p2_comps) // 2)
            elif p1_comps:
                selected = p1_comps
            else:
                selected = p2_comps
            
            offspring_components.extend([copy.deepcopy(c) for c in selected])
        
        return PromptGenome(
            genome_id="offspring",
            components=offspring_components,
            metadata={'crossover': True}
        )
    
    def _mutate(self, genome: PromptGenome) -> PromptGenome:
        """Mutate genome by modifying components."""
        mutated = copy.deepcopy(genome)
        
        if not mutated.components:
            return mutated
        
        # Choose mutation operation
        mutations = [
            self._mutate_swap_component,
            self._mutate_modify_text,
            self._mutate_add_component,
            self._mutate_remove_component
        ]
        
        mutation_fn = random.choice(mutations)
        mutated = mutation_fn(mutated)
        
        mutated.metadata['mutated'] = True
        return mutated
    
    def _mutate_swap_component(self, genome: PromptGenome) -> PromptGenome:
        """Swap two components of the same type."""
        if len(genome.components) < 2:
            return genome
        
        # Group by type
        by_type = {}
        for comp in genome.components:
            by_type.setdefault(comp.component_type, []).append(comp)
        
        # Find a type with multiple components
        swappable_types = [t for t, comps in by_type.items() if len(comps) >= 2]
        
        if not swappable_types:
            return genome
        
        swap_type = random.choice(swappable_types)
        comps = by_type[swap_type]
        
        # Swap two
        i, j = random.sample(range(len(comps)), 2)
        comps[i], comps[j] = comps[j], comps[i]
        
        return genome
    
    def _mutate_modify_text(self, genome: PromptGenome) -> PromptGenome:
        """Slightly modify a component's text."""
        if not genome.components:
            return genome
        
        comp = random.choice(genome.components)
        
        # Simple text mutations
        mutations = [
            lambda t: t.replace("must", "should"),
            lambda t: t.replace("avoid", "don't use"),
            lambda t: t + " Be concise.",
            lambda t: "Important: " + t
        ]
        
        mutation = random.choice(mutations)
        comp.text = mutation(comp.text)
        
        return genome
    
    def _mutate_add_component(self, genome: PromptGenome) -> PromptGenome:
        """Add a new component based on mined rules."""
        # Use mined rules to create new components
        if self.reflector.rules:
            rule = random.choice(self.reflector.rules)
            new_comp = TextComponent(
                name=f"rule_{rule.rule_id}",
                text=rule.action,
                component_type="guard",
                metadata={'from_rule': rule.rule_id}
            )
            genome.components.append(new_comp)
        
        return genome
    
    def _mutate_remove_component(self, genome: PromptGenome) -> PromptGenome:
        """Remove a random component (if not critical)."""
        if len(genome.components) <= 2:
            return genome
        
        # Don't remove system components
        removable = [c for c in genome.components if c.component_type != 'system']
        
        if removable:
            to_remove = random.choice(removable)
            genome.components.remove(to_remove)
        
        return genome
    
    def _get_best_genome(self) -> Optional[PromptGenome]:
        """Get best genome from archive."""
        if not self.archive:
            return None
        
        best = max(self.archive, key=lambda x: x['weighted_score'])
        return best['genome']

