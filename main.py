from parser import parser, parse_program

def main():
    file = open('parser_test.txt', 'r')
    text = file.read()
    file.close()

    lark_tree = parser.parse(text)
    syntax_tree = parse_program(lark_tree)

    print(str(syntax_tree))

if __name__ == '__main__':
    main()
