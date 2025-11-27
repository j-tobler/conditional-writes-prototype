from lark import Lark
from cfg import *


program_parser = Lark(
    r"""
    start:          _precondition procedure+
    _precondition:   "{" biconditional "}"
    procedure:      "proc" _SEP CNAME "{" block "}"
    block:          _statement*
    _statement:     assignment | conditional | loop | assume
    assignment:     CNAME ":=" biconditional ";"
    conditional:    "if" _SEP biconditional "{" block "}" ["else" "{" block "}"]
    loop:           "while" _SEP biconditional "{" block "}"
    assume:         "assume" _SEP biconditional ";"
    ?biconditional: implication | biconditional "<==>" implication
    ?implication:   disjunction | disjunction "==>" implication
    ?disjunction:   conjunction | disjunction "||" conjunction
    ?conjunction:   negation | conjunction "&&" negation
    ?negation:      parens_bool | "!" negation
    ?parens_bool:   inequality | "(" biconditional ")"
    ?inequality:    sum | sum (_inequality_op sum)*
    ?sum:           term | sum _sum_op term
    ?term:          signed_val | term _term_op signed_val
    ?signed_val:    parens_arith | _sum_op signed_val
    ?parens_arith:  id | num | "(" sum ")"
    id:             CNAME
    num:            INT
    !_sum_op:        "+" | "-"
    !_term_op:       "*"
    !_inequality_op: "<" | "<=" | "==" | ">=" | ">" | "!="
    _SEP:           WS+

    %import common (WS, CNAME, INT)
    %ignore WS
    """
)

def throw_parser_error():
    raise RuntimeError("Internal parser error.")

# start: procedure+
def parse_program(tree):
    return Program(parse_expr(tree.children[0]), [parse_procedure(proc) for proc in tree.children[1:]])

# procedure: "proc" _SEP CNAME "{" block "}"
def parse_procedure(tree):
    return Procedure(tree.children[0].value, parse_block(tree.children[1]))

# block: _statement*
def parse_block(tree):
    return [parse_statement(stmt) for stmt in tree.children]

# _statement: assignment | conditional | loop | assume
def parse_statement(tree):
    if tree.data.value == 'assignment':
        return parse_assignment(tree)
    elif tree.data.value == 'conditional':
        return parse_conditional(tree)
    elif tree.data.value == 'loop':
        return parse_loop(tree)
    elif tree.data.value == 'assume':
        return parse_assume(tree)
    raise throw_parser_error()

# assignment: CNAME ":=" biconditional ";"
def parse_assignment(tree):
    return Assignment(tree.children[0].value, parse_expr(tree.children[1]))

# conditional: "if" _SEP biconditional "{" block "}" ["else" "{" block "}"]
def parse_conditional(tree):
    expr = parse_expr(tree.children[0])
    if_branch = parse_block(tree.children[1])
    else_branch = [] if not tree.children[2] else parse_block(tree.children[2])
    return Conditional(expr, if_branch, else_branch)

# loop: "while" _SEP biconditional "{" block "}"
def parse_loop(tree):
    return Loop(parse_expr(tree.children[0]), parse_block(tree.children[1]))

# assume: "assume" _SEP biconditional ";"
def parse_assume(tree):
    return Assume(parse_expr(tree.children[0]))

"""
?biconditional: implication | biconditional "<==>" implication
?implication:   disjunction | disjunction "==>" implication
?disjunction:   conjunction | disjunction "||" conjunction
?conjunction:   negation | conjunction "&&" negation
?negation:      parens_bool | "!" negation
?parens_bool:   inequality | "(" biconditional ")"
?inequality:    sum | sum (inequality_op sum)*
?sum:           term | sum sum_op term
?term:          signed_val | term term_op signed_val
?signed_val:    parens_arith | sum_op signed_val
?parens_arith:  id | num | "(" sum ")"
id:             CNAME
num:            INT
"""
def parse_expr(tree):
    if tree.data.value == 'biconditional':
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[1]), BinOp.BICOND)
    
    elif tree.data.value == 'implication':
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[1]), BinOp.IMPL)
    
    elif tree.data.value == 'disjunction':
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[1]), BinOp.DISJ)
    
    elif tree.data.value == 'conjunction':
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[1]), BinOp.CONJ)
    
    elif tree.data.value == 'negation':
        return UnExpr(parse_expr(tree.children[0]), UnOp.NOT)
    
    elif tree.data.value == 'parens_bool':
        return parse_expr(tree.children[0])
    
    elif tree.data.value == 'inequality':
        def parse_inequality_op(op_str):
            if op_str == '<':
                return BinOp.LT
            elif op_str == '<=':
                return BinOp.LE
            elif op_str == '==':
                return BinOp.EQ
            elif op_str == '>=':
                return BinOp.GE
            elif op_str == '>':
                return BinOp.GT
            elif op_str == '!=':
                return BinOp.NE
            else:
                throw_parser_error()
        inequality_op = parse_inequality_op(tree.children[1].value)
        conjunction = BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[2]), inequality_op)
        i = 2
        while i + 1 != len(tree.children):
            inequality_op = parse_inequality_op(tree.children[i+1].value)
            inequality = BinExpr(parse_expr(tree.children[i]), parse_expr(tree.children[i+2]), inequality_op)
            conjunction = BinExpr(conjunction, inequality, BinOp.CONJ)
            i += 2
        return conjunction
    
    elif tree.data.value == 'sum':
        op = None
        if tree.children[1].value == '+':
            op = BinOp.PLUS
        elif tree.children[1].value == '-':
            op = BinOp.MINUS
        else:
            throw_parser_error()
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[2]), op)
    
    elif tree.data.value == 'term':
        op = None
        if tree.children[1].value == '*':
            op = BinOp.TIMES
        else:
            throw_parser_error()
        return BinExpr(parse_expr(tree.children[0]), parse_expr(tree.children[2]), op)
    
    elif tree.data.value == 'signed_val':
        op = None
        if tree.children[0].value == '+':
            op = UnOp.POS
        elif tree.children[0].value == '-':
            op = UnOp.NEG
        else:
            throw_parser_error()
        return UnExpr(parse_expr(tree.children[1]), op)
    
    elif tree.data.value == 'parens_arith':
        return parse_expr(tree.children[0])
    
    elif tree.data.value == 'id':
        return tree.children[0].value
    
    elif tree.data.value == 'num':
        return int(tree.children[0].value)

    else:
        throw_parser_error()
