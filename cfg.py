from enum import Enum


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
        return '\n\n'.join([p.pretty(indent) for p in self.procedures])

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
        d = I.capture_interference(I, D, d, r)
        self.pre = d
        # compute guar
        g |= I.transitions(D, d, self)
        # compute post-state
        d = D.transfer_assign(d, self)
        return d, r, g

class Conditional(Lang):
    def __init__(self, condition, if_branch: list, else_branch: list):
        self.condition = condition
        self.if_branch = if_branch
        self.else_branch = else_branch
        self.pre = None

    def pretty(self, indent=0):
        else_str = ''
        if self.else_branch:
            else_str = ' else {\n' + '\n'.join([s.pretty(indent + 1) for s in self.else_branch]) + '\n' + '\t' * indent + '}'
        if_header = '\t' * indent + 'if ' + str(self.condition) + ' {\n'
        if_str = if_header + '\n'.join([s.pretty(indent + 1) for s in self.if_branch]) + '\n' + '\t' * indent + '}'
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        return pre + '\n' + if_str + else_str

    def analyse(self, D, I, d, r, g):
        # stabilise d
        d = I.capture_interference(I, D, d, r)
        self.pre = d
        # init results for true branch
        d_true = D.transfer_filter(d, self.condition)
        r_true = r
        g_true = g
        # init results for false branch
        d_false = D.transfer_filter(d, UnExpr(self.condition, UnOp.NOT))
        r_false = r
        g_false = g
        # compute results
        for stmt in self.if_branch:
            d_true, r_true, g_true = stmt.analyse(D, I, d_true, r_true, g_true)
        for stmt in self.else_branch:
            d_false, r_false, g_false = stmt.analyse(D, I, d_false, r_false, g_false)
        # merge branches
        d = d_true | d_false
        r = r_true | r_false
        g = g_true | g_false
        return d, r, g

class Loop(Lang):
    def __init__(self, condition, statements: list):
        self.condition = condition
        self.statements = statements
        self.pre = None

    def pretty(self, indent=0):
        header = '\t' * indent + 'while ' + str(self.condition) + ' {\n'
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        return pre + '\n' + header + '\n'.join([s.pretty(indent + 1) for s in self.statements]) + '\n' + '\t' * indent + '}'

    def analyse(self, D, I, d, r, g):
        while True:
            initial_d = d
            initial_r = r
            initial_g = g
            # stabilise d
            d = I.capture_interference(I, D, d, r)
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
        d = I.capture_interference(I, D, d, r)
        self.pre = d # record loop invariant as loop precondition
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
        d = I.capture_interference(I, D, d, r)
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
