import argparse
import cv2 as cv

from Programs.check import run_check
from Programs.utils import afficher

def main():
    parser = argparse.ArgumentParser(description="PCB wiring checking program")

    subparsers = parser.add_subparsers(dest="command")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check the wiring of a given pcb")
    check_parser.add_argument("--input", required=True, help="File name of the picture")
    check_parser.add_argument("--verbose", action="store_true")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show the last output again")

    args = parser.parse_args()

    if args.command == "check":
        run_check(args.input)
    elif args.command == "show":
        img = cv.imread("result.jpg")
        afficher(img)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()