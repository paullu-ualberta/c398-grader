import json
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
import logging

from omr import mark_file
from pathlib import Path


logger = logging.getLogger("CLI")
logger.setLevel(logging.INFO)


def mark_and_write_graded(answer_file_path, attempt_file_path, /, output_fname_pattern):
    answer_file = Path(answer_file_path)
    with (
        answer_file.open("rb") as answer_file,
        attempt_file_path.open("rb") as attempt_file,
    ):
        score, total_score, marked_file_content = mark_file(
            attempt_file.read(), answer_file.read()
        )

    output_fname = output_fname_pattern.replace("%F", attempt_file_path.name)
    output_file = attempt_file_path.parent / output_fname
    with output_file.open("wb+") as f:
        f.write(marked_file_content)
    return score, total_score


def write_to_json_out(scores_dict, json_file_name):
    with open(json_file_name, "w+") as f:
        logger.debug(f"Writing dict {scores_dict}")
        json.dump(scores_dict, f, indent=4)


def do_mark(args):
    to_mark = Path(args.to_mark).resolve()
    files_to_mark = list(to_mark.glob("*.pdf")) if to_mark.is_dir() else (to_mark,)
    pool = Pool(args.threads)
    marker = partial(
        mark_and_write_graded, args.answer_file, output_fname_pattern=args.out_fname_pat
    )
    results = pool.map(marker, files_to_mark)
    results_dict = dict(
        (str(f.name), score) for f, (score, _) in zip(files_to_mark, results)
    )
    write_to_json_out(results_dict, args.out_file)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("to_mark")
    mark_parser.add_argument("answer_file")
    mark_parser.add_argument("-j", "--threads", type=int, default=1)
    mark_parser.add_argument("-o", "--output", dest="out_file", default="scores.json")
    mark_parser.add_argument(
        "-p", "--graded-file-name", default="graded_%F.pdf", dest="out_fname_pat"
    )

    args = parser.parse_args()
    if args.command == "mark":
        do_mark(args)


if __name__ == "__main__":
    main()
