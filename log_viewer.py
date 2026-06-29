import argparse
import json
import math


def print_json_object(data, spaces=0):
    if type(data) is list:
        print(f"{' ' * spaces}[")
    else:
        print(f"{' ' * spaces}{{")

    for key in data:
        if type(data) is dict:
            print(f"{' ' * (spaces + 2)}", end="")
            print(f"\033[1m\033[34m{key}\033[0m", end=": ")
            value = data[key]
        else:
            value = key

        if type(value) is str:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

        if type(value) is dict or type(value) is list:
            print_json_object(value, spaces + 2)
        elif type(value) is str:
            print(f'"{value}"')
        else:
            print(f"{value}")
    if type(data) is list:
        print(f"{' ' * spaces}]")
    else:
        print(f"{' ' * spaces}}}")


def main():
    parser = argparse.ArgumentParser(
        description="Show the Json data in a pretty format"
    )
    parser.add_argument("jsonl_file", help="Jsonl file with the log data")
    parser.add_argument(
        "--num",
        type=int,
        help="show n first log messages",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="filter by message type",
    )
    parser.add_argument(
        "--show_events",
        action="store_true",
        help="show the different type of events",
    )

    line_num = 0
    args = parser.parse_args()
    jsonl_file = args.jsonl_file
    to_show = math.inf
    if args.num:
        to_show = args.num
    filter_type = None
    if args.filter:
        filter_type = args.filter
    events = set()
    show_events = args.show_events

    with open(jsonl_file, "r") as f:
        for line in f.readlines():
            line_num += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if show_events:
                events.add(data["event"])
                continue

            if filter_type and data["event"] != filter_type:
                continue

            print_json_object(data)

            if line_num >= to_show:
                break

    if show_events:
        print(events)


if __name__ == "__main__":
    main()
