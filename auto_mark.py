from omr import mark_file

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("file")
    mark_parser.add_argument("answer_file")
    args = parser.parse_args()

    if args.command == "mark":
        with (
            open(args.file, "rb") as attempt_file,
            open(args.answer_file, "rb") as answer_file,
        ):
            mark_file(attempt_file.read(), answer_file.read())


if __name__ == "__main__":
    main()
