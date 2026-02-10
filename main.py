# Copyright 2026 The Commonwealth of Australia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from analysis import *
from parsing import program_parser, parse_program
from cfg import Program
from config import Config, StateDomains
from stats import Stats


def main():
    # parse args and initialise global configuration struct Config
    Config.init()

    # read file containing target program
    file = open(Config.target_path, 'r')
    text = file.read()
    file.close()

    # parse target program with lark parser
    lark_tree = program_parser.parse(text)
    # parse lark AST into our custom AST representing a CFG
    program = parse_program(lark_tree)

    # configure interference domains and state domains
    I = ConditionalWritesDomain
    if Config.state_domain == StateDomains.CONSTANTS:
        D = ConstantDomain
    elif Config.state_domain == StateDomains.DISJUNCTIVE_CONSTANTS:
        D = DisjunctiveConstantsDomain
    else:
        raise RuntimeError('Could not find the abstract domain: ' + str(Config.state_domain))

    # begin analysis
    Stats.state_lattice_joins = 0
    Stats.state_lattice_meets = 0
    Stats.start_timer()
    G, R = run_analysis(I, D, program)
    Stats.end_timer()

    if not Stats.is_configured():
        raise RuntimeError(f'Stats block was not fully configured at the end of analysis:\n{Stats.to_str()}')

    if Config.out:
        file = open(Config.out, 'a')
        file.write(Config.to_str() + '\n')
        file.write(Stats.to_str() + '\n\n')
        file.close()
    else:
        # display resulting RG conditions and performance stats
        # note that 'program' maintains its own proof outline during the analysis
        print_results(G, R, program)


def print_results(guars: dict, relys: dict, program: Program):
    print('Stable Proof Outlines:')
    print()
    print(str(program))
    print()
    print('Guarantee Conditions:')
    print('\n'.join([f'{proc.name}:\n{guar}' for proc, guar in guars.items()]))
    print()
    print('Rely Conditions:')
    print('\n'.join([f'{proc.name}:\n{rely}' for proc, rely in relys.items()]))
    print()
    print(f'Iterations: {Stats.iterations}')
    print(f'State lattice join and meet operations count: {Stats.state_lattice_joins + Stats.state_lattice_meets}')
    print(f'Analysis time: {Stats.performance_time}')
    print(f'Verified: {Stats.verified}')


def run_analysis(I: type[InterferenceDomain], D: type[AbstractDomain], program: Program):
    def generate_rely(proc, guars):
        rely = I.bot(D)
        for other_proc in guars:
            if proc == other_proc:
                continue
            rely |= guars[other_proc]
        if Config.transitive_mode:
            rely = I.close(D, rely)
        return rely
    
    # initialise RG conditions
    guars = {proc: I.bot(D) for proc in program.procedures}
    relys = {}
    # derive the program precondition once, to be used repeatedly in later iterations
    pre = D.transfer_filter(D.top(), program.precondition)
    iterations = 0
    while True:
        iterations += 1
        old_guars = guars.copy()
        for proc in program.procedures:
            relys[proc] = generate_rely(proc, guars)
            guars[proc] = proc.analyse(D, I, pre, relys[proc], I.bot(D))[2]
        if guars == old_guars:
            break
    Stats.iterations = iterations
    Stats.verified = program.is_verified()
    return guars, relys


if __name__ == '__main__':
    main()
