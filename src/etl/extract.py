"""Extract: Load raw data into DuckDB."""

import sys
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager
from src.utils.io import PathManager, SchemaRegistry, load_config

console = Console()


def extract_data(config_path: str = "configs/project.yaml") -> Dict[str, Any]:
    """
    Extract data from CSV files into DuckDB warehouse.
    
    Args:
        config_path: Path to project configuration file
        
    Returns:
        Dictionary with extraction results and statistics
    """
    console.print("\n[bold blue]>> Starting Data Extraction[/bold blue]\n")
    
    # Load configuration
    config = load_config(config_path)
    path_mgr = PathManager()
    
    # Initialize DuckDB
    db_path = config['project'].get('warehouse_path', 'data/warehouse/bi.duckdb')
    db = DuckDBManager(db_path)
    
    # Schema registry
    schema_reg = SchemaRegistry(Path(config_path))
    
    results = {
        'tables_loaded': [],
        'row_counts': {},
        'errors': []
    }
    
    # Extract each dataset
    datasets = config['project'].get('datasets', [])
    
    for dataset_path in datasets:
        dataset_path = Path(dataset_path)
        table_name = dataset_path.stem  # e.g., orders.csv -> orders
        
        try:
            console.print(f"  >> Loading [cyan]{dataset_path.name}[/cyan]...", end=" ")
            
            if not dataset_path.exists():
                console.print(f"[yellow][WARN] File not found, skipping[/yellow]")
                results['errors'].append(f"{dataset_path} not found")
                continue
            
            # Load CSV into DuckDB
            db.load_csv(dataset_path, table_name)
            
            # Get row count
            row_count = db.get_row_count(table_name)
            results['row_counts'][table_name] = row_count
            results['tables_loaded'].append(table_name)
            
            console.print(f"[green][OK] {row_count:,} rows[/green]")
            
        except Exception as e:
            console.print(f"[red][FAIL] Error: {e}[/red]")
            results['errors'].append(f"{table_name}: {str(e)}")
    
    # Display summary table
    if results['tables_loaded']:
        console.print("\n[bold green]>> Extraction Summary[/bold green]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Columns", justify="right")
        
        for tbl in results['tables_loaded']:
            info = db.get_table_info(tbl)
            table.add_row(
                tbl,
                f"{results['row_counts'][tbl]:,}",
                str(len(info))
            )
        
        console.print(table)
    
    if results['errors']:
        console.print("\n[bold red][WARN] Errors:[/bold red]")
        for error in results['errors']:
            console.print(f"  • {error}")
    
    db.close()
    console.print("\n[bold green][SUCCESS] Extraction Complete[/bold green]\n")
    
    return results


if __name__ == "__main__":
    extract_data()

