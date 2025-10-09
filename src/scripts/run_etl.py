"""CLI script to run ETL pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.etl.extract import extract_data
from src.etl.load import load_marts
from rich.console import Console

console = Console()


def main():
    """Run complete ETL pipeline."""
    console.print("[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]  BI-DSPy-GEPA: ETL Pipeline          [/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")
    
    try:
        # Extract
        console.print("[bold]Step 1: Extract[/bold]")
        extract_results = extract_data()
        
        # Load/Transform
        console.print("\n[bold]Step 2: Transform & Load[/bold]")
        load_results = load_marts()
        
        # Summary
        console.print("\n[bold green]========================================[/bold green]")
        console.print("[bold green]  ETL Complete!                        [/bold green]")
        console.print("[bold green]========================================[/bold green]\n")
        
        console.print(f"Tables loaded: {len(extract_results['tables_loaded'])}")
        console.print(f"Marts created: {len(load_results['marts_created'])}")
        console.print(f"Parquet exports: {len(load_results['exports'])}")
        
        if extract_results['errors'] or load_results['errors']:
            console.print(f"\n[yellow]Warnings: {len(extract_results['errors']) + len(load_results['errors'])} issue(s)[/yellow]")
        
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red][FAIL] ETL Failed:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

