import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    'filename',
    type=str,
    metavar='FILE',
    help='path to the target program')

arg_parser.add_argument(
    '-t',
    action='store_true',
    help='enable transitive analysis mode (omit for non-transitive mode)')

arg_parser.add_argument(
    '-n',
    type=int,
    metavar='INT',
    help='stabilisation precision metric (higher is more precise); omit for maximum precision')


def main():
    args = arg_parser.parse_args()
    print(args.t)

if __name__ == '__main__':
    main()
