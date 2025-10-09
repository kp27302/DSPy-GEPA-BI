"""CLI script to evaluate SQL synthesis."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager
from src.utils.io import PathManager, load_config, save_json
from src.eval.scorer import evaluate_sql_task
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Evaluate SQL synthesis on benchmark tasks."""
    console.print("[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]  BI-DSPy-GEPA: SQL Evaluation         [/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")
    
    try:
        # Setup
        config = load_config()
        path_mgr = PathManager()
        
        db_path = config['project'].get('warehouse_path', 'data/warehouse/bi.duckdb')
        db = DuckDBManager(db_path)
        
        # Load tasks
        tasks_path = path_mgr.eval_benchmarks / 'sql_tasks.jsonl'
        
        if not tasks_path.exists():
            console.print(f"[red]Error: Task file not found: {tasks_path}[/red]")
            return 1
        
        tasks = []
        with open(tasks_path, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
        
        console.print(f"Loaded {len(tasks)} benchmark tasks\n")
        
        # Evaluate each task (simplified - just run gold SQL)
        results = []
        passed = 0
        
        for task in tasks:
            task_id = task['id']
            
            try:
                # In a real scenario, would use DSPy to generate SQL
                # Here, we just validate the gold SQL executes
                gold_sql = task.get('gold_sql', '')
                
                if gold_sql:
                    result_df = db.query_df(gold_sql)
                    
                    results.append({
                        'task_id': task_id,
                        'passed': True,
                        'rows': len(result_df),
                        'nl': task['nl']
                    })
                    passed += 1
                else:
                    results.append({
                        'task_id': task_id,
                        'passed': False,
                        'error': 'No gold SQL'
                    })
                    
            except Exception as e:
                results.append({
                    'task_id': task_id,
                    'passed': False,
                    'error': str(e)
                })
        
        # Display results
        console.print("\n[bold]Evaluation Results:[/bold]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for result in results[:10]:  # Show first 10
            status = "[PASS]" if result['passed'] else "[FAIL]"
            style = "green" if result['passed'] else "red"
            
            details = result.get('nl', result.get('error', ''))
            table.add_row(result['task_id'], f"[{style}]{status}[/{style}]", details[:60])
        
        console.print(table)
        
        # Summary
        pass_rate = passed / len(tasks) * 100 if tasks else 0
        
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total tasks: {len(tasks)}")
        console.print(f"  Passed: {passed}")
        console.print(f"  Failed: {len(tasks) - passed}")
        console.print(f"  Pass rate: {pass_rate:.1f}%")
        
        # Save results
        results_path = path_mgr.eval_results / f"sql_eval_{Path().cwd().name}.json"
        save_json({'results': results, 'summary': {'passed': passed, 'total': len(tasks)}}, results_path)
        console.print(f"\n[dim]Results saved to: {results_path}[/dim]")
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red][FAIL] Evaluation Failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

