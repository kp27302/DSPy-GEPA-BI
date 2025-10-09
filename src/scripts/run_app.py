"""CLI script to run Streamlit dashboard."""

import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console

console = Console()


def main():
    """Launch Streamlit dashboard application."""
    console.print("[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]  BI-DSPy-GEPA: Dashboard Launch       [/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")
    
    console.print("Starting Streamlit dashboard...\n")
    
    # Path to dashboard app
    app_path = Path(__file__).parent.parent / "app" / "dashboard_app.py"
    
    if not app_path.exists():
        console.print(f"[red]Error: Dashboard app not found at {app_path}[/red]")
        return 1
    
    try:
        # Launch streamlit
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red][FAIL] Failed to launch dashboard:[/bold red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

