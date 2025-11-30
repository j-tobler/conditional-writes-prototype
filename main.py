from analysis import *
from parsing import program_parser, parse_program
from cfg import Program
from config import Config, StateDomains, set_config
from stats import Stats


def main():
    # parse args and initialise global configuration struct Config
    set_config()

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
    Stats.start_timer()
    G, R = run_analysis(I, D, program)
    Stats.end_timer()

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


def run_analysis(I: InterferenceDomain, D: AbstractDomain, program: Program):
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
    return guars, relys


if __name__ == '__main__':
    main()
