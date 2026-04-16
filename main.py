import argparse
import cv2 as cv
import json

from Programs.check import run_check
from Programs.utils import afficher

with open("Configuration/config.json", "r") as f:
            config = json.load(f)

def main():
    parser = argparse.ArgumentParser(description="PCB wiring checking program")

    subparsers = parser.add_subparsers(dest="command")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check the wiring of a given pcb")
    check_parser.add_argument("--input", required=True, help="File name of the picture")
    check_parser.add_argument("--iref", required=False, help="Custom iref numbers = 4 ints separated by ','")
    check_parser.add_argument("--verbose", action="store_true")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show the last output again")

    args = parser.parse_args()

    if args.command == "check":
        path = args.input
        if not(config["pictures_folder"] in path) :
            path = config["pictures_folder"] + "/" + path
        if not(config["suffix_after_bonding"] in path) :
            path = path + "_" + config["suffix_after_bonding"]
        if not(config["pictures_format"] in path) :
            path = path + config["pictures_format"]
        if args.iref != None :
            trim_nb_str = args.iref.split(",")
            trim_nb_int = []
            for x in trim_nb_str :
                trim_nb_int.append(int(x))
            run_check(path, iref = trim_nb_int)
        else :
            run_check(path)
    elif args.command == "show":
        img = cv.imread("result.jpg")
        afficher(img)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()