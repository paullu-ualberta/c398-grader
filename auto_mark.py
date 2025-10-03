import json
import csv
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
import logging

from omr import mark_file, mark_single_file
from pathlib import Path


logger = logging.getLogger("CLI")
logger.setLevel(logging.INFO)


def build_output_path(attempt_file_path, output_fname_pattern):
    output_fname = output_fname_pattern.replace("%F", attempt_file_path.name)
    return attempt_file_path.parent / output_fname


def mark_and_write_graded(answer_file_path, attempt_file_path, /, output_fname_pattern):
    answer_file = Path(answer_file_path)
    with answer_file.open("rb") as answer_file:
        with attempt_file_path.open("rb") as attempt_file:
            score, total_score, marked_file_content = mark_file(
                attempt_file.read(), answer_file.read()
            )

    output_file = build_output_path(attempt_file_path, output_fname_pattern)
    with output_file.open("wb+") as f:
        f.write(marked_file_content)
    return score, total_score


def write_scores_to_outfile(scores_dict, outfile_name):
    outfile = Path(outfile_name).resolve()
    logger.debug(f"Writing dict {scores_dict}")
    with outfile.open("w+") as f:
        if outfile_name.endswith(".json"):
            json.dump(scores_dict, f, indent=4)
        else:
            writer = csv.writer(f)
            writer.writerows(scores_dict.items())


def do_mark_single_file(args, file_to_mark):
    answer_file = Path(args.answer_file).resolve()
    with file_to_mark.open("rb") as attempt_file:
        with answer_file.open("rb") as answer_file:
            marked_file_content = mark_single_file(
                attempt_file.read(), answer_file.read()
            )

    output_file = build_output_path(file_to_mark, args.out_fname_pat)
    with output_file.open("wb+") as f:
        f.write(marked_file_content)


def do_mark(args):
    to_mark = Path(args.to_mark).resolve()
    if to_mark.is_dir() and args.single_file:
        print(
            "Argument --single-file only makes sense when a file is being marked",
            str(to_mark),
            "is a directory",
        )
        exit(1)
    files_to_mark = list(to_mark.glob("*.pdf")) if to_mark.is_dir() else (to_mark,)
    if not args.single_file:
        pool = Pool(args.threads)
        marker = partial(
            mark_and_write_graded,
            args.answer_file,
            output_fname_pattern=args.out_fname_pat,
        )
        results = pool.map(marker, files_to_mark)
        results_dict = dict(
            (str(f.name), score) for f, (score, _) in zip(files_to_mark, results)
        )
        write_scores_to_outfile(results_dict, args.out_file)
    else:
        assert len(files_to_mark) == 1
        file_to_mark = files_to_mark[0]
        do_mark_single_file(args, file_to_mark)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("to_mark")
    mark_parser.add_argument("answer_file")
    mark_parser.add_argument("-j", "--threads", type=int, default=1)
    mark_parser.add_argument("-o", "--output", dest="out_file", default="scores.csv")
    mark_parser.add_argument(
        "-p", "--graded-file-name", default="graded_%F", dest="out_fname_pat"
    )
    mark_parser.add_argument(
        "--single-file", default=False, action="store_true", dest="single_file"
    )

    args = parser.parse_args()
    if args.command == "mark":
        do_mark(args)


if __name__ == "__main__":
    main()
