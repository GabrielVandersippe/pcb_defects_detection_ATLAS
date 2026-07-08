from rich.console import Console
from rich.table import Table
from rich.align import Align
from rich import box
import time
import atexit
import sys

import json

console = Console(record=True)

def write_log():
    command = " ".join([sys.executable, *sys.argv])
    with open("pcb_defect_detector.log", "w", encoding="utf-8") as f:
        f.write(f"{command}\n")
        f.write(console.export_text())

atexit.register(write_log)

def show_config(config):
    console.rule(f"[bold red]{"CONFIG"}[/bold red]", characters="=")
    table = Table(show_header=False, box=box.ASCII)
    for key,value in config.items():
        table.add_row(f"[bold blue]{key} [/bold blue]", f"[cyan]{value}[cyan]")
    console.print(Align.center(table))
    console.print("\n")

def print_success(msg):
    console.print(f"[bold green]  ✔  [/bold green][green]{msg}")

def print_error(msg):
    console.print(f"[bold red]  ✘  [/bold red][red]{msg}")

def print_info(msg):
    console.print(f"[blue]{msg}[blue]")

def afficher_bilan(wire_nb, expected_wire_nb, short_nb_left_not_crit, short_nb_right_not_crit, short_nb_left_crit, short_nb_right_crit, nb_wrong_track, nb_wrong_pad, iref_th, iref_read, missing_wires, crit_short_wires_left, non_crit_short_wires_left,crit_short_wires_right, non_crit_short_wires_right, crit_endpoints_track, crit_endpoints_pad):
    with open("../config/config.json", "r") as f:
            config = json.load(f)
    lang = config["language"]
    
    console.print("\n")
    if lang == "en" :
        console.rule(f"[bold red]{"SUMMARY"}[/bold red]", characters="=")
    else :
        console.rule(f"[bold red]{"BILAN"}[/bold red]", characters="=")
    table = Table(show_header=False, box=box.ASCII)

    if lang == "en" :
        table.add_row( ("[bold green] ✔ " if wire_nb == expected_wire_nb else "[bold red] ✘ ") + "Number of wires", 
                  f"[bold]{wire_nb}/{expected_wire_nb}", 
                  f"{list(set(missing_wires))}",
                  f"[bold green]OK" if wire_nb == expected_wire_nb else "[bold red]NOK")
    
        table.add_row(("[bold green] ✔ " if short_nb_left_not_crit == 0 else "[bold dark_orange] ? ")+"Non-critical short circuits (left)", 
                  f"[bold]{short_nb_left_not_crit}",
                  f"{list(set(non_crit_short_wires_left))}", 
                  "[bold green]OK" if short_nb_left_not_crit == 0 else "[bold dark_orange] ~ ")
    
        table.add_row(("[bold green] ✔ " if short_nb_left_crit == 0 else "[bold red] ✘ ")+"Critical short circuits (left)",
                  f"[bold]{short_nb_left_crit}",
                  f"{list(set(crit_short_wires_left))}",
                  "[bold green]OK" if short_nb_left_crit == 0 else "[bold red]NOK")    
    
        table.add_row(("[bold green] ✔ " if short_nb_right_not_crit == 0 else "[bold dark_orange] ? ")+"Non-critical short circuits (right)",
                  f"[bold]{short_nb_right_not_crit}",
                  f"{list(set(non_crit_short_wires_right))}",
                  "[bold green]OK" if short_nb_right_not_crit == 0 else "[bold dark_orange] ~ ")
    
        table.add_row(("[bold green] ✔ " if short_nb_right_crit == 0 else "[bold red] ✘ ")+"Critical short circuits (right)",
                f"[bold]{short_nb_right_crit}",
                f"{list(set(crit_short_wires_right))}",
                "[bold green]OK" if short_nb_right_crit == 0 else "[bold red]NOK")
    
        table.add_row(("[bold green] ✔ " if nb_wrong_track == 0 else "[bold red] ✘ ")+"Improperly wired wires on tracks (except short circuits) ", 
                  f"[bold]{nb_wrong_track}",
                  f"{list(set(crit_endpoints_track))}", 
                  "[bold green]OK" if nb_wrong_track == 0 else "[bold red]NOK")
        
        table.add_row(("[bold green] ✔ " if nb_wrong_pad == 0 else "[bold red] ✘ ")+"Improperly wired wires on pads (except short circuits) ", 
                  f"[bold]{nb_wrong_pad}",
                  f"{list(set(crit_endpoints_pad))}", 
                  "[bold green]OK" if nb_wrong_pad == 0 else "[bold red]NOK")
        
        table.add_row(("[bold green] ✔ " if tuple(iref_th) == tuple(iref_read) else "[bold red] ✘ ")+"IREF (GA1,GA2,GA3,GA4) ", 
                  f"expected : [bold]({iref_th[0]}, {iref_th[1]}, {iref_th[2]}, {iref_th[3]})",
                  f"read : [bold]({iref_read[0]}, {iref_read[1]}, {iref_read[2]}, {iref_read[3]})", 
                  "[bold green]OK" if tuple(iref_th) == tuple(iref_read) else "[bold red]NOK")
    else :
        table.add_row( ("[bold green] ✔ " if wire_nb == expected_wire_nb else "[bold red] ✘ ") +  "Nombre de fils", 
                  f"[bold]{wire_nb}/{expected_wire_nb}",
                  f"{list(set(missing_wires))}", 
                  f"[bold green]OK" if wire_nb == expected_wire_nb else "[bold red]NOK")
    
        table.add_row(("[bold green] ✔ " if short_nb_left_not_crit == 0 else "[bold dark_orange] ? ")+"Courts-circuits (gauche) non critiques", 
                  f"[bold]{short_nb_left_not_crit}",
                  f"{list(set(non_crit_short_wires_left))}", 
                  "[bold green]OK" if short_nb_left_not_crit == 0 else "[bold dark_orange] ~ ")
    
        table.add_row(("[bold green] ✔ " if short_nb_left_crit == 0 else "[bold red] ✘ ")+"Courts-circuits (gauche) critiques",
                  f"[bold]{short_nb_left_crit}",
                  f"{list(set(crit_short_wires_left))}",
                  "[bold green]OK" if short_nb_left_crit == 0 else "[bold red]NOK")    
    
        table.add_row(("[bold green] ✔ " if short_nb_right_not_crit == 0 else "[bold dark_orange] ? ")+"Courts-circuits (droite) non critiques",
                  f"[bold]{short_nb_right_not_crit}",
                  f"{list(set(non_crit_short_wires_right))}",
                  "[bold green]OK" if short_nb_right_not_crit == 0 else "[bold dark_orange] ~ ")
    
        table.add_row(("[bold green] ✔ " if short_nb_right_crit == 0 else "[bold red] ✘ ")+"Courts-circuits (droite) critiques",
                f"[bold]{short_nb_right_crit}",
                f"{list(set(crit_short_wires_right))}",
                "[bold green]OK" if short_nb_right_crit == 0 else "[bold red]NOK")
        
        table.add_row(("[bold green] ✔ " if nb_wrong_track == 0 else "[bold red] ✘ ")+"Fils mal câblés sur les pistes (hors court-circuits) ", 
                  f"[bold]{nb_wrong_track}",
                  f"{list(set(crit_endpoints_track))}", 
                  "[bold green]OK" if nb_wrong_track == 0 else "[bold red]NOK")
        
        table.add_row(("[bold green] ✔ " if nb_wrong_pad == 0 else "[bold red] ✘ ")+"Fils mal câblés sur les pads (hors court-circuits) ", 
                  f"[bold]{nb_wrong_pad}",
                  f"{list(set(crit_endpoints_pad))}", 
                  "[bold green]OK" if nb_wrong_pad == 0 else "[bold red]NOK")
        
        table.add_row(("[bold green] ✔ " if tuple(iref_th) == tuple(iref_read) else "[bold red] ✘ ")+"IREF (GA1,GA2,GA3,GA4) ", 
                  f"attendu : [bold]({iref_th[0]}, {iref_th[1]}, {iref_th[2]}, {iref_th[3]})",
                  f"lu : [bold]({iref_read[0]}, {iref_read[1]}, {iref_read[2]}, {iref_read[3]})", 
                  "[bold green]OK" if tuple(iref_th) == tuple(iref_read) else "[bold red]NOK")
    
    console.print(Align.center(table))
    console.print("\n")
    time.sleep(2)