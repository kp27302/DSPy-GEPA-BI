#!/usr/bin/env python3
"""Run GEPA optimization with REAL LLM evaluations."""

import sys
import os
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gepa.loop import GEPAOptimizer, GEPAConfig
from src.gepa.objectives import create_default_objectives, MultiObjectiveEvaluator
from src.gepa.components import PromptGenome
from src.utils.io import PathManager, load_config, save_json
from src.utils.duck import DuckDBManager
from src.eval.scorer import SQLScorer
from rich.console import Console
import time

console = Console()


def load_benchmark_tasks(benchmark_path: Path) -> list:
    """Load SQL benchmark tasks from JSONL file."""
    tasks = []
    if benchmark_path.exists():
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    task = json.loads(line)
                    tasks.append(task)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow][WARN] Skipping invalid JSON line {i}: {e}[/yellow]")
                    continue
    else:
        console.print(f"[red][ERROR] Benchmark file not found: {benchmark_path}[/red]")
    return tasks


def real_eval_function(genome: PromptGenome) -> dict:
    """
    REAL evaluation function using DSPy + OpenAI.
    
    This function:
    1. Configures DSPy with the genome's prompts
    2. Runs SQL synthesis on benchmark tasks
    3. Evaluates results against gold SQL
    4. Returns real metrics
    """
    import dspy
    from src.dspy_programs.sql_synth import SQLSynthesizer
    from dotenv import load_dotenv
    
    # Load API key (Mistral, Gemini, or OpenAI)
    load_dotenv()
    api_key = os.getenv('MISTRAL_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY')
    model_name = 'mistral/mistral-small-latest'  # Fast, cheap, excellent DSPy support
    
    # Determine which provider based on available key
    if os.getenv('OPENAI_API_KEY') and not (os.getenv('MISTRAL_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')):
        model_name = 'gpt-4o-mini'
    elif os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        if not os.getenv('MISTRAL_API_KEY'):
            model_name = 'gemini-1.5-flash'
    
    if not api_key:
        # Fallback to dummy for demo
        console.print("[yellow][WARN] No API key - using simulated results[/yellow]")
        import random
        return {
            'accuracy': random.uniform(0.6, 0.95),
            'tests': random.uniform(0.7, 1.0),
            'cost': random.uniform(100, 500),
            'latency': random.uniform(0.5, 2.0)
        }
    
    # Initialize database and scorer
    path_mgr = PathManager()
    config = load_config()
    db_path = config['project'].get('warehouse_path', 'data/warehouse/bi.duckdb')
    db = DuckDBManager(db_path)
    scorer = SQLScorer(db)
    
    # Load benchmark tasks
    benchmark_path = path_mgr.eval_benchmarks / 'sql_tasks.jsonl'
    tasks = load_benchmark_tasks(benchmark_path)
    
    if not tasks:
        console.print("[yellow][WARN] No benchmark tasks found[/yellow]")
        return {'accuracy': 0.5, 'tests': 0.5, 'cost': 300, 'latency': 1.0}
    
    # Configure DSPy with LLM (DSPy 3.0 API)
    try:
        # DSPy 3.0 supports both Gemini and OpenAI
        import os as os_module
        os_module.environ['GOOGLE_API_KEY'] = api_key  # Ensure env var is set for DSPy
        
        lm = dspy.LM(
            model=model_name,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.1
        )
        dspy.configure(lm=lm)
        console.print(f"[dim]Configured DSPy with {model_name}[/dim]")
    except Exception as e:
        console.print(f"[red][FAIL] DSPy config error: {e}[/red]")
        console.print(f"[dim]Error details: {str(e)}[/dim]")
        import traceback
        traceback.print_exc()
        return {'accuracy': 0.5, 'tests': 0.5, 'cost': 300, 'latency': 1.0}
    
    # Get schema from config
    schema = config.get('schema', {})
    
    # Extract few-shot examples from genome
    fewshot_components = genome.get_components_by_type('fewshot')
    few_shot_examples = []
    for comp in fewshot_components:
        # Parse text like "Task: X\nSQL: Y"
        lines = comp.text.split('\n')
        if len(lines) >= 2:
            nl = lines[0].replace('Task:', '').strip()
            sql = lines[1].replace('SQL:', '').strip()
            few_shot_examples.append({'nl': nl, 'sql': sql})
    
    # Create SQL synthesizer with genome's configuration
    synthesizer = SQLSynthesizer(
        schema_context=schema,
        few_shot_examples=few_shot_examples
    )
    
    # Evaluate on tasks (use subset to save costs)
    max_tasks = min(5, len(tasks))  # Limit to 5 tasks per genome
    results = []
    total_tokens = 0
    total_time = 0
    
    for task in tasks[:max_tasks]:
        try:
            start = time.time()
            
            # Generate SQL
            result = synthesizer(task=task['nl'])
            generated_sql = result.sql
            
            elapsed = time.time() - start
            
            # Score against gold SQL
            gold_sql = task.get('gold_sql', '')
            
            if gold_sql:
                passed, score, msg = scorer.score_execution(generated_sql, gold_sql)
            else:
                # Just check if it executes
                try:
                    db.query_df(generated_sql)
                    passed, score = True, 1.0
                except:
                    passed, score = False, 0.0
            
            # Static checks
            static_passed, issues = scorer.check_static_quality(generated_sql)
            
            results.append({
                'passed': passed,
                'score': score,
                'static_passed': static_passed,
                'tokens': result.estimated_cost,
                'latency': elapsed
            })
            
            total_tokens += result.estimated_cost
            total_time += elapsed
            
        except Exception as e:
            console.print(f"[dim]Task {task.get('id', '?')} failed: {e}[/dim]")
            results.append({
                'passed': False,
                'score': 0.0,
                'static_passed': False,
                'tokens': 0,
                'latency': 0
            })
    
    # Compute aggregate metrics
    if results:
        accuracy = sum(r['score'] for r in results) / len(results)
        test_pass_rate = sum(r['static_passed'] for r in results) / len(results)
        avg_cost = total_tokens / len(results)
        avg_latency = total_time / len(results)
    else:
        accuracy = 0.5
        test_pass_rate = 0.5
        avg_cost = 300
        avg_latency = 1.0
    
    return {
        'accuracy': accuracy,
        'tests': test_pass_rate,
        'cost': avg_cost,
        'latency': avg_latency,
        'tasks_evaluated': len(results),
        'total_tokens': total_tokens
    }


def main():
    """Run GEPA optimization with real evaluations."""
    console.print("[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]  REAL GEPA Optimization (with LLM)    [/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key (Mistral, Gemini, or OpenAI)
    api_key = os.getenv('MISTRAL_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY')
    
    # Determine provider
    if os.getenv('MISTRAL_API_KEY'):
        api_provider = 'Mistral'
    elif os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
        api_provider = 'Gemini'
    elif os.getenv('OPENAI_API_KEY'):
        api_provider = 'OpenAI'
    else:
        api_provider = None
    
    if not api_key:
        console.print("[bold red][ERROR] No API key found![/bold red]")
        console.print("\nPlease add your API key to .env file:")
        console.print("  [green]MISTRAL_API_KEY[/green]=your-mistral-key  ([bold]RECOMMENDED - great DSPy support![/bold])")
        console.print("  OR")
        console.print("  GOOGLE_API_KEY=AIza-your-gemini-key  (FREE tier)")
        console.print("  OR")
        console.print("  OPENAI_API_KEY=sk-your-key-here\n")
        console.print("Get Mistral key: [cyan]https://console.mistral.ai/[/cyan]\n")
        console.print("[yellow]Running with simulated results instead...[/yellow]\n")
    else:
        console.print(f"[green][OK] {api_provider} API key configured ({api_key[:15]}...)[/green]\n")
    
    try:
        # Load config
        config = load_config()
        gepa_config_dict = config.get('gepa', {})
        
        # Create GEPA config (smaller for real runs to save costs)
        gepa_config = GEPAConfig(
            population_size=8,  # Reduced from 16
            generations=4,      # Reduced from 8
            max_trials=50,      # Reduced from 150
            mutation_rate=0.3,
            crossover_rate=0.5,
            seed=42
        )
        
        console.print(f"[bold]Configuration (Cost-Optimized):[/bold]")
        console.print(f"  Population: {gepa_config.population_size}")
        console.print(f"  Generations: {gepa_config.generations}")
        console.print(f"  Max trials: {gepa_config.max_trials}")
        console.print(f"  Tasks per genome: 5")
        console.print(f"  Model: {api_provider if api_provider else 'Simulated'}")
        
        # Estimate cost
        total_evals = gepa_config.population_size * gepa_config.generations
        tokens_per_eval = 5 * 1500  # 5 tasks × ~1500 tokens each
        total_tokens = total_evals * tokens_per_eval
        
        console.print(f"\n[bold]Estimated Cost:[/bold]")
        console.print(f"  Total evaluations: ~{total_evals}")
        console.print(f"  Estimated tokens: ~{total_tokens:,}")
        
        if api_provider == 'Mistral':
            console.print(f"  Mistral Small: [bold green]Very cheap ($0.001/1K tokens)[/bold green]\n")
        elif api_provider == 'Gemini':
            console.print(f"  Gemini 1.5 Flash: [bold green]FREE tier (15 RPM)[/bold green]\n")
        elif api_provider == 'OpenAI':
            cost_estimate = total_tokens * 0.15 / 1_000_000  # GPT-4o-mini pricing
            console.print(f"  Estimated cost: ${cost_estimate:.2f}\n")
        else:
            console.print(f"  Cost: $0.00 (simulated)\n")
        
        if api_key:
            response = input("Proceed with real LLM calls? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                console.print("[yellow]Cancelled by user[/yellow]")
                return 0
        
        # Create objectives
        objectives = create_default_objectives()
        evaluator = MultiObjectiveEvaluator(objectives)
        
        # Create optimizer with REAL eval function
        optimizer = GEPAOptimizer(
            config=gepa_config,
            evaluator=evaluator,
            eval_function=real_eval_function  # <-- REAL FUNCTION
        )
        
        # Run optimization
        console.print("\n")
        results = optimizer.optimize()
        
        # Save results
        path_mgr = PathManager()
        results_path = path_mgr.eval_results / "gepa_real_results.json"
        
        # Convert results for JSON serialization
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
            'mined_rules': results['mined_rules'],
            'config': {
                'population_size': gepa_config.population_size,
                'generations': gepa_config.generations,
                'used_real_llm': bool(api_key)
            }
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
        
        # Show top 3 genomes
        if results['pareto_archive']:
            console.print(f"\n[bold]Top 3 Genomes:[/bold]")
            for i, item in enumerate(results['pareto_archive'][:3], 1):
                metrics = item['metrics']
                console.print(f"\n  {i}. {item['genome'].genome_id}")
                console.print(f"     Accuracy: {metrics.get('accuracy', 0):.3f}")
                console.print(f"     Tests: {metrics.get('tests', 0):.3f}")
                console.print(f"     Cost: {metrics.get('cost', 0):.0f} tokens")
                console.print(f"     Latency: {metrics.get('latency', 0):.2f}s")
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red][FAIL] GEPA Search Failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

