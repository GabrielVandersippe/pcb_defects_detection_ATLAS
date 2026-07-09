import argparse
import cv2 as cv
import json
import numpy as np

from programs.check import run_check
from programs.utils import afficher, find_suffix
from programs.debug_tools import magnifying_glass_final_result
from programs.output import show_config

cv.setNumThreads(1)
cv.ocl.setUseOpenCL(False)

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description="PCB wiring checking program")

    subparsers = parser.add_subparsers(dest="command")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check the wiring of a given pcb")
    check_parser.add_argument("--input", required=True, help="File name of the picture")
    check_parser.add_argument("--iref", required=False, help="Custom iref numbers = 4 ints separated by ','")
    check_parser.add_argument("--verbose", type=int, choices=[0,1,2,3], help="Change the level of detail of the output. Goes from 0 to 3. Default : 0.")
    check_parser.add_argument("--aggressiveness", choices=["low", "medium", "high"], help="Change the aggressiveness of the algorithm. Stronger aggressiveness means more false positives.")
    check_parser.add_argument("--override-targets", action="store_true", help="Override automatic check of the targets for manual definition.")
    check_parser.add_argument("--skip-pulltest", action="store_true", help="Skip the pulltest wires in the checking.")
    check_parser.add_argument("--read-corners", action="store_true", help="Read the corners position from a json file.")
    check_parser.add_argument("--read-targets", action="store_true", help="Read the targets position from a json file.")
    # Show command
    show_parser = subparsers.add_parser("show", help="Show the last output again")

    # Edit Config command
    config_parser = subparsers.add_parser("config", help="Edit the current configuration file.")
    config_parser.add_argument("--folder", help="Change the folder inside which the module pictures are located.")
    config_parser.add_argument("--format", help="Change the format of the pictures. Default : '.jpg'")
    config_parser.add_argument("--language", choices=['fr', 'en'], help="Change the language : 'fr' or 'en'.")
    config_parser.add_argument("--suffix-after", help="Change the suffix that comes before the reference of the module after bonding.")
    config_parser.add_argument("--suffix-before", help="Change the suffix that comes before the reference of the module before bonding.")
    config_parser.add_argument("--verbose", type=int, choices=[0,1,2,3], help="Change the level of detail of the output. Goes from 0 to 3. Default : 0.")
    config_parser.add_argument("--zoom-select", type=positive_int, help="Change the zooming power of the magnifying glass function for the selection (integer). Default : 4")
    config_parser.add_argument("--zoom-visualize", type=positive_int, help="Change the zooming power of the magnifying glass function for the visualization (integer). Default : 2")
    config_parser.add_argument("--show", action="store_true", help="Show the current configuration.")

    args = parser.parse_args()

    if args.command == "check":        
        with open("../config/config.json", "r") as f:
                    config = json.load(f)
                    
        path = args.input
        pic_folder = config["pictures_folder"]
        suffixes = config["suffix_after_bonding"]
        pic_format = config["pictures_format"]
        suffix_in_path = False
        for s in suffixes :
            if s in path :
                suffix_in_path = True
        if not(suffix_in_path) :
            path = find_suffix(path, suffixes, pic_folder)
        if not(pic_folder in path) :
            path = pic_folder + "/" + path
        if not(pic_format in path) :
            path = path + pic_format
        if args.aggressiveness:
            config["aggressiveness_level"] = args.aggressiveness
        if args.iref != None :
            trim_nb_str = args.iref.split(",")
            trim_nb_int = []
            for x in trim_nb_str :
                trim_nb_int.append(int(x))
            run_check(path, iref = trim_nb_int, verbose=args.verbose, config=config, override_targets=args.override_targets, skip_pulltest=args.skip_pulltest, read_corners=args.read_corners, read_targets=args.read_targets)
        else :
            run_check(path, verbose=args.verbose, config=config, override_targets=args.override_targets, skip_pulltest=args.skip_pulltest, read_corners=args.read_corners, read_targets=args.read_targets)

    elif args.command == "show":
        with open("../temp/path.txt", "r") as f:
            path = f.readline()
        if path[-1] == "/" :
            file_name = (path.split("/")[-2]).split(".")[0]
        else :
            file_name = (path.split("/")[-1]).split(".")[0]
        magnifying_glass_final_result(path, file_name)

    elif args.command == "config":
        with open("../config/config.json", "r") as f:
            config = json.load(f)

        if args.folder:
            assert()
            config["pictures_folder"] = args.folder
        if args.format:
            config["pictures_format"] = args.format            
        if args.language:
            config["language"] = args.language            
        if args.suffix_after:
            config["suffix_after_bonding"] = args.suffix_after            
        if args.suffix_before:
            config["suffix_before_bonding"] = args.suffix_before            
        if args.verbose is not None:
            config["verbose"] = int(args.verbose)            
        if args.zoom_select is not None:
            config["zoom_select"] = int(args.zoom_select)
        if args.zoom_visualize is not None:
            config["zoom_visualize"] = int(args.zoom_visualize)

        if args.show:
            print("Configuration mise à jour.") if config["language"]=='fr' else print("Updated config.")
            show_config(config)
        else :
            print("Configuration mise à jour. Exécuter config --show pour l'afficher.") if config["language"]=='fr' else print("Updated config. Run config --show to show.")

        with open("../config/config.json", "w") as f:
            json.dump(config, f, indent=4)


    else:
        parser.print_help()

if __name__ == "__main__":
    main()