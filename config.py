import argparse
from enum import Enum


class StateDomains(Enum):
    CONSTANTS = 0
    DISJUNCTIVE_CONSTANTS = 1

state_domain_names = {
    'constants': StateDomains.CONSTANTS,
    'disjunctive constants': StateDomains.DISJUNCTIVE_CONSTANTS
}

class Config:
    target_path: str = None
    state_domain: StateDomains = None
    transitive_mode: bool = None
    precision: int = None

    @staticmethod
    def is_configured():
        return None not in [
            Config.target_path,
            Config.state_domain,
            Config.transitive_mode,
            Config.precision
        ]

def set_config():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        'filename',
        type=str,
        metavar='FILE',
        help='path to the target program')

    arg_parser.add_argument(
        'state_domain',
        choices=list(state_domain_names.keys()),
        help='the abstract domain for representing states')

    arg_parser.add_argument(
        '-t',
        action='store_true',
        help='enable transitive analysis mode (omit for non-transitive mode)')

    arg_parser.add_argument(
        '-n',
        type=int,
        metavar='INT',
        help='stabilisation precision metric (higher is more precise); omit for maximum precision')

    args = arg_parser.parse_args()
    Config.target_path = args.filename
    Config.state_domain = state_domain_names[args.state_domain]
    Config.transitive_mode = args.t
    Config.precision = -1 if args.n == None else args.n
    if not Config.is_configured():
        return RuntimeError('Failed to configure analysis.')
