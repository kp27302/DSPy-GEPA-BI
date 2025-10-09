"""Load: Execute transformations and materialize data marts."""

import sys
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager
from src.utils.io import PathManager, load_config

console = Console()


def load_marts(config_path: str = "configs/project.yaml",
               transform_sql_path: str = "src/etl/transform.sql") -> Dict[str, Any]:
    """
    Execute transformation SQL and materialize data marts.
    
    Args:
        config_path: Path to project configuration
        transform_sql_path: Path to transformation SQL file
        
    Returns:
        Dictionary with load results and statistics
    """
    console.print("\n[bold blue]>> Loading Data Marts[/bold blue]\n")
    
    # Load configuration
    config = load_config(config_path)
    path_mgr = PathManager()
    
    # Initialize DuckDB
    db_path = config['project'].get('warehouse_path', 'data/warehouse/bi.duckdb')
    db = DuckDBManager(db_path)
    
    results = {
        'marts_created': [],
        'row_counts': {},
        'exports': [],
        'errors': []
    }
    
    # Read transformation SQL
    transform_sql = Path(transform_sql_path).read_text()
    
    # Remove comment lines
    lines = [line for line in transform_sql.split('\n') if line.strip() and not line.strip().startswith('--')]
    clean_sql = '\n'.join(lines)
    
    # Split into individual statements
    statements = [s.strip() for s in clean_sql.split(';') if s.strip()]
    
    # Execute transformations with progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Executing transformations...", total=len(statements))
        
        for stmt in statements:
            try:
                # Extract table name from CREATE TABLE statement
                table_name = None
                stmt_upper = stmt.upper()
                if 'CREATE' in stmt_upper and 'TABLE' in stmt_upper:
                    # Handle "CREATE OR REPLACE TABLE table_name"
                    import re
                    match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(\w+)', stmt_upper)
                    if match:
                        table_name = match.group(1).lower()
                
                # Execute statement
                db.execute(stmt)
                
                if table_name and db.table_exists(table_name):
                    row_count = db.get_row_count(table_name)
                    results['marts_created'].append(table_name)
                    results['row_counts'][table_name] = row_count
                    console.print(f"  [OK] Created [cyan]{table_name}[/cyan]: [green]{row_count:,} rows[/green]")
                
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"  [red][FAIL] Error executing statement: {e}[/red]")
                results['errors'].append(str(e))
                progress.update(task, advance=1)
    
    # Export marts to Parquet
    console.print("\n[bold blue]>> Exporting to Parquet[/bold blue]\n")
    
    for mart in results['marts_created']:
        try:
            parquet_path = path_mgr.data_warehouse / f"{mart}.parquet"
            db.export_parquet(mart, parquet_path)
            results['exports'].append(str(parquet_path))
            console.print(f"  [OK] Exported [cyan]{mart}[/cyan] -> {parquet_path.name}")
        except Exception as e:
            console.print(f"  [red][FAIL] Error exporting {mart}: {e}[/red]")
            results['errors'].append(f"Export {mart}: {str(e)}")
    
    db.close()
    
    if results['errors']:
        console.print("\n[bold red][WARN] Errors:[/bold red]")
        for error in results['errors']:
            console.print(f"  • {error}")
    
    console.print("\n[bold green][SUCCESS] Load Complete[/bold green]\n")
    
    return results


if __name__ == "__main__":
    load_marts()

