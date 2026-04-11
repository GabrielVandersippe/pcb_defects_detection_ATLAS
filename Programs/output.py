from rich.console import Console
from rich.table import Table
from rich.align import Align
from rich.live import Live
import time

console = Console()

def print_success(msg):
    console.print(f"[bold green]{"  ○  " + msg}[/bold green]")

def print_error(msg):
    console.print(f"[bold red]{"  ○  " + msg}[/bold red]")

def print_info(msg):
    console.print(f"[bold blue]{msg}[/bold blue]")

def afficher_bilan(wire_nb, short_nb_left_not_crit, short_nb_right_not_crit, nb_wrong_track):
    console.rule(f"[bold red]{"BILAN"}[/bold red]")
    time.sleep(1)
    table = Table(show_header=False)

    with Live(Align.center(table), refresh_per_second=4):
        table.add_row("[green]Fils                                  ", f"[bold]{wire_nb}", f"[bold green]OK")
        time.sleep(1)
        table.add_row(("[green]" if short_nb_left_not_crit == 0 else "[red]")+"Courts-circuits (gauche) non critiques", f"[bold]{short_nb_left_not_crit}", "[bold green]OK" if short_nb_left_not_crit == 0 else "[bold red]NOK")
        time.sleep(1)
        table.add_row(("[green]" if short_nb_right_not_crit == 0 else "[red]")+"Courts-circuits (droite) non critiques", f"[bold]{short_nb_right_not_crit}", "[green]OK" if short_nb_right_not_crit == 0 else "[bold red]NOK")
        time.sleep(1)
        table.add_row(("[green]" if nb_wrong_track == 0 else "[red]")+"Fils mal câblés (hors court-circuits) ", f"[bold]{nb_wrong_track}", "[bold green]OK" if nb_wrong_track == 0 else "[bold red]NOK")
        time.sleep(2)