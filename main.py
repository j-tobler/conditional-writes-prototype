from analysis import *

def main():
    file = open('parser_test.txt', 'r')
    text = file.read()
    file.close()

    lark_tree = program_parser.parse(text)
    program = parse_program(lark_tree)

    for proc in program.procedures:
        proc.analyse(ConstantDomain, ConditionalWritesDomain, ConstantDomain.top(), ConditionalWritesDomain.bot(), ConditionalWritesDomain.bot())

    print(str(program))

if __name__ == '__main__':
    main()
