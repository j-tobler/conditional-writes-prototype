from lark import Lark

parser = Lark(
    r"""
    start:          procedure+
    procedure:      "proc" _SEP CNAME "{" _statement* "}"
    _statement:     assignment | conditional | loop | assume
    assignment:     CNAME ":=" expr ";"
    conditional:    "if" _SEP expr "{" _statement* "}"
    loop:           "while" _SEP expr "{" _statement* "}"
    assume:         "assume" _SEP expr ";"
    ?expr:          implication | expr "<==>" implication
    ?implication:   disjunction | disjunction "==>" implication
    ?disjunction:   conjunction | disjunction "||" conjunction
    ?conjunction:   negation | conjunction "&&" negation
    ?negation:      atom | "!" negation
    ?atom:          sum | "(" expr ")"
    ?sum:           term | sum sum_op term
    ?term:          signed_val | term term_op signed_val
    ?signed_val:    val | sum_op signed_val
    val:            CNAME | INT | "(" sum ")"
    id:             CNAME
    num:            INT
    !sum_op:        "+" | "-"
    !term_op:       "*" | "/" | "%"
    _SEP:           WS+

    %import common (WS, CNAME, INT)
    %ignore WS
    """
)

if __name__ == '__main__':
    text = "proc main { james := 5 + 6 + 7; if cond {} james := 6 + 7; }"
    tree = parser.parse(text)
    print(tree.pretty())
