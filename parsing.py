from lark import Lark
from enum import Enum
from analysis import ANALYSIS_MODE

program_parser = Lark(
    r"""
    start:          precondition procedure+
    precondition:   "{" biconditional "}"
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

class BinOp(Enum):
    BICOND = 0
    IMPL = 1
    DISJ = 2
    CONJ = 3
    LT = 4
    LE = 5
    EQ = 6
    GE = 7
    GT = 8
    NE = 9
    PLUS = 10
    MINUS = 11
    TIMES = 12

bool_op_str = ["<==>", "==>", "||", "&&", "<", "<=", "==", ">=", ">", "!=", "+", "-", "*"]

class UnOp(Enum):
    NOT = 0
    POS = 1
    NEG = 2

un_op_str = ["!", "+", "-"]

class Lang:
    def pretty(self, indent=0):
        raise NotImplementedError('Function \'pretty\' not implemented.')

    def __str__(self):
        return self.pretty()

class Program(Lang):
    def __init__(self, precondition, procedures: list):
        self.precondition = precondition
        self.procedures = procedures

    def pretty(self, indent=0):
        pre_str = '\t' * indent + '{' + str(self.precondition) + '}'
        return pre_str + '\n\n' + '\n\n'.join([p.pretty(indent) for p in self.procedures])

class Procedure(Lang):
    def __init__(self, name: str, statements: list):
        self.name = name
        self.statements = statements

    def pretty(self, indent=0):
        return '\t' * indent + 'proc ' + self.name + ' {\n' + '\n'.join([s.pretty(indent + 1) for s in self.statements]) + '\n' + '\t' * indent + '}'

    def analyse(self, D, I, d, r, g):
        for stmt in self.statements:
            d, r, g = stmt.analyse(D, I, d, r, g)
        return d, r, g

class Assignment(Lang):
    def __init__(self, lhs: str, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.pre = None

    def pretty(self, indent=0):
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        return pre + '\n' + '\t' * indent + str(self.lhs) + ' := ' + str(self.rhs) + ';'

    def analyse(self, D, I, d, r, g):
        # stabilise d
        if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
            d = I.stabilise(D, d, r, PRECISION)
        else:
            while True:
                old_d = d
                d = I.stabilise(D, d, r, PRECISION)
                if d == old_d:
                    break
        self.pre = d
        # compute guar
        g = g | I.transitions(D, d, self)
        # compute post-state
        d = D.transfer_assign(d, self)
        return d, r, g

class Conditional(Lang):
    def __init__(self, condition, if_branch: list, else_branch: list):
        self.condition = condition
        self.if_branch = if_branch
        self.else_branch = else_branch

    def pretty(self, indent=0):
        else_str = ''
        if self.else_branch:
            else_str = ' else {\n' + '\n'.join([s.pretty(indent + 1) for s in self.else_branch]) + '\n' + '\t' * indent + '}'
        if_header = '\t' * indent + 'if ' + str(self.condition) + ' {\n'
        if_str = if_header + '\n'.join([s.pretty(indent + 1) for s in self.if_branch]) + '\n' + '\t' * indent + '}'
        return if_str + else_str

    def analyse(self, D, I, d, r, g):
        # stabilise d
        if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
            d = I.stabilise(D, d, r, PRECISION)
        else:
            while True:
                old_d = d
                d = I.stabilise(D, d, r, PRECISION)
                if d == old_d:
                    break
        d_true = D.transfer_filter(d, self.condition)
        r_true = r
        g_true = g
        d_false = D.transfer_filter(d, UnExpr(self.condition, UnOp.NOT))
        r_false = r
        g_false = g
        for stmt in self.if_branch:
            d_true, r_true, g_true = stmt.analyse(D, I, d_true, r_true, g_true)
        for stmt in self.else_branch:
            d_false, r_false, g_false = stmt.analyse(D, I, d_false, r_false, g_false)
        d = d_true | d_false
        r = r_true | r_false
        g = g_true | g_false
        return d, r, g

class Loop(Lang):
    def __init__(self, condition, statements: list):
        self.condition = condition
        self.statements = statements

    def pretty(self, indent=0):
        header = '\t' * indent + 'while ' + str(self.condition) + ' {\n'
        return header + '\n'.join([s.pretty(indent + 1) for s in self.statements]) + '\n' + '\t' * indent + '}'

    def analyse(self, D, I, d, r, g):
        while True:
            initial_d = d
            initial_r = r
            initial_g = g
            # stabilise d
            if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
                d = I.stabilise(D, d, r, PRECISION)
            else:
                while True:
                    old_d = d
                    d = I.stabilise(D, d, r, PRECISION)
                    if d == old_d:
                        break
            # filter d
            d = D.transfer_filter(d, self.condition)
            # analyse loop body
            for stmt in self.statements:
                d, r, g = stmt.analyse(D, I, d, r, g)
            # join to current invariant
            d |= initial_d
            r |= initial_r
            g |= initial_g
            if d == initial_d and r == initial_r and g == initial_g:
                break
        # stabilise d
        if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
            d = I.stabilise(D, d, r, PRECISION)
        else:
            while True:
                old_d = d
                d = I.stabilise(D, d, r, PRECISION)
                if d == old_d:
                    break
        # filter not self.condition
        d = D.transfer_filter(d, UnExpr(self.condition, UnOp.NOT))
        return d, r, g

class Assume(Lang):
    def __init__(self, condition):
        self.condition = condition
        self.pre = None

    def pretty(self, indent=0):
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        return pre + '\n' + '\t' * indent + 'assume ' + str(self.condition) + ';'

    def analyse(self, D, I, d, r, g):
        # stabilise d
        if ANALYSIS_MODE == AnalysisMode.TRANSITIVE:
            d = I.stabilise(D, d, r, PRECISION)
        else:
            while True:
                old_d = d
                d = I.stabilise(D, d, r, PRECISION)
                if d == old_d:
                    break
        self.pre = d
        # filter d
        d = D.transfer_filter(d, self.condition)
        return d, r, g

class BinExpr(Lang):
    def __init__(self, lhs, rhs, op: BinOp):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def pretty(self, indent=0):
        lhs_str = str(self.lhs)
        if not isinstance(self.lhs, str) and not isinstance(self.lhs, int):
            lhs_str = '(' + lhs_str + ')'
        rhs_str = str(self.rhs)
        if not isinstance(self.rhs, str) and not isinstance(self.rhs, int):
            rhs_str = '(' + rhs_str + ')'
        return '\t' * indent + lhs_str + ' ' + bool_op_str[self.op.value] + ' ' + rhs_str

class UnExpr(Lang):
    def __init__(self, rhs, op: UnOp):
        self.rhs = rhs
        self.op = op

    def pretty(self, indent=0):
        return '\t' * indent + un_op_str[self.op.value] + '(' + str(self.rhs) + ')'

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
