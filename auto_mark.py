from omr import mark_file
from pathlib import Path

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial


def read_mark_and_write(attempt_file_path, answer_file_path):
    answer_file = Path(answer_file_path)
    with answer_file.open("rb") as answer_file:
        mark_and_write(answer_file.read(), attempt_file_path)


def mark_and_write(answer_file_content, attempt_file_path):
    print(f"Marking {attempt_file_path}")
    attempt_file = Path(attempt_file_path)
    with attempt_file.open("rb") as f:
        marked_file = mark_file(f.read(), answer_file_content)
    out_file = Path(attempt_file_path) / ".." / f"graded_{attempt_file.name}"
    out_file = out_file.resolve()
    print(f"Writing to {out_file}")
    with out_file.open("wb+") as f:
        f.write(marked_file)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("file")
    mark_parser.add_argument("answer_file")

    mark_dir_parser = subparsers.add_parser("mark-dir")
    mark_dir_parser.add_argument("dir")
    mark_dir_parser.add_argument("answer_file")
    mark_dir_parser.add_argument("-j", "--threads", type=int, default=1)

    args = parser.parse_args()
    if args.command == "mark":
        read_mark_and_write(args.file, args.answer_file)
    elif args.command == "mark-dir":
        dir = Path(args.dir).resolve()
        if not dir.is_dir():
            print(f"{dir} is not a directory")
            exit(1)
        pool = Pool(args.threads)
        answer_file = Path(args.answer_file)
        with answer_file.open("rb") as f:
            answer_file_content = f.read()

        marker = partial(mark_and_write, answer_file_content)
        pool.map(marker, dir.glob("*.pdf"))


if __name__ == "__main__":
    main()
