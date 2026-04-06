from rich.console import Console

console = Console()

def print_success(msg):
    console.print(f"[bold green]{msg}[/bold green]")

def print_error(msg):
    console.print(f"[bold red]{msg}[/bold red]")

def print_info(msg):
    console.print(f"[bold blue]{msg}[/bold blue]")