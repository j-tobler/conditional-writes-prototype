from analysis import *
import sys

def main():
    global ANALYSIS_MODE
    global PRECISION
    global CONSTANT_LATTICE_CALLS

    file = open(sys.argv[1], 'r')
    text = file.read()
    file.close()

    lark_tree = program_parser.parse(text)
    program = parse_program(lark_tree)

    I = ConditionalWritesDomain
    D = ConstantDomain

    pre = D.top() # todo: parameterise
    guars = {proc: I.bot() for proc in program.procedures}
    posts = {proc: D.top() for proc in program.procedures}
    iterations = 0
    while True:
        iterations += 1
        old_guars = guars.copy()
        for proc in program.procedures:
            rely = I.bot()
            for other_proc in guars:
                if proc == other_proc:
                    continue
                rely |= guars[other_proc]
            if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
                rely = I.close(D, rely)
            d, r, g = proc.analyse(D, I, pre, rely, guars[proc])
            guars[proc] = g
            # stabilise d
            if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
                d = I.stabilise(D, d, r, PRECISION)
            else:
                while True:
                    old_d = d
                    d = I.stabilise(D, d, r, PRECISION)
                    if d == old_d:
                        break
            posts[proc] = d
        if guars == old_guars:
            break
    final_post = D.top();
    for post in posts.values():
        final_post &= post

    print(str(program))
    print()
    print('Local Postconditions:')
    for proc, post in posts.items():
        print(proc.name + ': ' + str(post))
    print()
    print('Guarantee Conditions:')
    for proc, guar in guars.items():
        print(proc.name + ':\n' + str(guar))
        print()
    print('Program Postcondition: ' + str(final_post))
    print('Number of state-lattice join and meet operations: ' + str(CONSTANT_LATTICE_CALLS))
    print(f'Iterations: {iterations}')


if __name__ == '__main__':
    main()
