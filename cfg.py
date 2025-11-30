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
            d, r, g = stmt.analyse(D, I, d, r, g, False)
        return d, r, g

class Assignment(Lang):
    def __init__(self, lhs: list, rhs: list):
        self.lhs = lhs
        self.rhs = rhs
        self.pre = None

    def pretty(self, indent=0):
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        lhs = ', '.join(self.lhs)
        rhs = ', '.join([str(expr) for expr in self.rhs])
        return pre + '\n' + '\t' * indent + lhs + ' := ' + rhs + ';'

    def analyse(self, D, I, d, r, g, is_atomic):
        if not is_atomic:
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

    def analyse(self, D, I, d, r, g, is_atomic):
        if not is_atomic:
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
            d_true, r_true, g_true = stmt.analyse(D, I, d_true, r_true, g_true, is_atomic)
        for stmt in self.else_branch:
            d_false, r_false, g_false = stmt.analyse(D, I, d_false, r_false, g_false, is_atomic)
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

    def analyse(self, D, I, d, r, g, is_atomic):
        while True:
            initial_d = d
            initial_r = r
            initial_g = g
            if not is_atomic:
                # stabilise d
                d = I.capture_interference(I, D, d, r)
            # filter d
            d = D.transfer_filter(d, self.condition)
            # analyse loop body
            for stmt in self.statements:
                d, r, g = stmt.analyse(D, I, d, r, g, is_atomic)
            # join to current invariant
            d |= initial_d
            r |= initial_r
            g |= initial_g
            if d == initial_d and r == initial_r and g == initial_g:
                break
        if not is_atomic:
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

    def analyse(self, D, I, d, r, g, is_atomic):
        if not is_atomic:
            # stabilise d
            d = I.capture_interference(I, D, d, r)
        self.pre = d
        # filter d
        d = D.transfer_filter(d, self.condition)
        return d, r, g

class Atomic(Lang):
    def __init__(self, statements: list):
        self.statements = statements
        self.pre = None

    def pretty(self, indent=0):
        header = '\t' * indent + 'atomic {\n'
        pre = '\t' * indent + '{' + str(self.pre) + '}'
        return pre + '\n' + header + '\n'.join([s.pretty(indent + 1) for s in self.statements]) + '\n' + '\t' * indent + '}'

    def analyse(self, D, I, d, r, g, is_atomic):
        if is_atomic:
            raise RuntimeError('Cannot have an atomic block within another atomic block.')
        # stabilise d
        d = I.capture_interference(I, D, d, r)
        self.pre = d
        # analyse atomic block
        for stmt in self.statements:
            d, r, g = stmt.analyse(D, I, d, r, g, True)
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


def to_dnf(expr):
    if is_atom(expr):
        return expr
    elif isinstance(expr, BinExpr):
        if expr.op == BinOp.BICOND:
            left_implies_right = BinExpr(lhs, rhs, BinOp.IMPL)
            right_implies_left = BinExpr(rhs, lhs, BinOp.IMPL)
            conjunction = BinExpr(left_implies_right, right_implies_left, BinOp.CONJ)
            return to_dnf(conjunction)
        if expr.op == BinOp.IMPL:
            neg_lhs = UnExpr(expr.lhs, UnOp.NOT)
            disjunction = BinExpr(neg_lhs, rhs, BinOp.OR)
            return to_dnf(disjunction)
        if expr.op == BinOp.DISJ:
            return BinExpr(to_dnf(expr.lhs), to_dnf(expr.rhs), BinOp.DISJ)
        if expr.op == BinOp.CONJ:
            lhs_dnf = to_dnf(expr.lhs)
            rhs_dnf = to_dnf(expr.rhs)
            # args are now of type: atom, negation, disjunction, conjunction
            if isinstance(lhs_dnf, BinExpr) and lhs_dnf.op == BinOp.DISJ:
                # distribute conjunction over disjunction
                new_lhs = BinExpr(lhs_dnf.lhs, rhs_dnf, BinOp.CONJ)
                new_rhs = BinExpr(lhs_dnf.rhs, rhs_dnf, BinOp.CONJ)
                return to_dnf(BinExpr(new_lhs, new_rhs, BinOp.DISJ))
            if isinstance(rhs_dnf, BinExpr) and rhs_dnf.op == BinOp.DISJ:
                # distribute conjunction over disjunction
                new_lhs = BinExpr(lhs_dnf, rhs_dnf.lhs, BinOp.CONJ)
                new_rhs = BinExpr(lhs_dnf, rhs_dnf.rhs, BinOp.CONJ)
                return to_dnf(BinExpr(new_lhs, new_rhs, BinOp.DISJ))
            return BinExpr(lhs_dnf, rhs_dnf, BinOp.CONJ)
        raise RuntimeError('This BinExpr was not identified as an atom, though it is not handled by to_dnf:\n' + str(expr))
    elif isinstance(expr, UnExpr):
        if expr.op == UnOp.NOT:
            neg = apply_negation(expr.rhs)
            if isinstance(neg, UnExpr) and neg.op == UnOp.NOT:
                return neg
            return to_dnf(neg)
        return expr
    else:
        raise RuntimeError('Could not convert this to DNF because it is not an expression:\n' + str(expr))


def apply_negation(expr):
    # ensures: if the result is a NOT, the body must be an atom
    if is_atom(expr):
        if isinstance(expr, BinExpr):
            if expr.op == BinOp.LT:
                return BinExpr(expr.lhs, expr.rhs, BinOp.GE)
            if expr.op == BinOp.LE:
                return BinExpr(expr.lhs, expr.rhs, BinOp.GT)
            if expr.op == BinOp.EQ:
                return BinExpr(expr.lhs, expr.rhs, BinOp.NE)
            if expr.op == BinOp.GE:
                return BinExpr(expr.lhs, expr.rhs, BinOp.LT)
            if expr.op == BinOp.GT:
                return BinExpr(expr.lhs, expr.rhs, BinOp.LE)
            if expr.op == BinOp.NE:
                return BinExpr(expr.lhs, expr.rhs, BinOp.EQ)
        return UnOp(expr, UnOp.NOT)
    if isinstance(expr, BinExpr):
        # we have expr.op in [BinOp.BICOND, BinOp.IMPL, BinOp.DISJ, BinOp.CONJ]
        if expr.op == BinOp.BICOND:
            # convert !(a <==> b) to (a && !b) || (!a && b)
            neg_lhs = apply_negation(expr.lhs)
            neg_rhs = apply_negation(expr.rhs)
            lhs_disjunct = BinExpr(expr.lhs, neg_rhs, BinOp.CONJ)
            rhs_disjunct = BinExpr(neg_lhs, expr.rhs, BinOp.CONJ)
            return BinExpr(lhs_disjunct, rhs_disjunct, BinOp.DISJ)
        if expr.op == BinOp.IMPL:
            # convert !(a ==> b) to a && !b
            neg_rhs = apply_negation(expr.rhs)
            return BinExpr(expr.lhs, neg_rhs, BinOp.CONJ)
        if expr.op == BinOp.DISJ:
            # convert !(a || b) to !a && !b
            neg_lhs = apply_negation(expr.lhs)
            neg_rhs = apply_negation(expr.rhs)
            return BinExpr(neg_lhs, neg_rhs, BinOp.CONJ)
        if expr.op == BinOp.CONJ:
            # convert !(a && b) to !a || !b
            neg_lhs = apply_negation(expr.lhs)
            neg_rhs = apply_negation(expr.rhs)
            return BinExpr(neg_lhs, neg_rhs, BinOp.DISJ)
        raise RuntimeError('Unexpected atom representing a BinExpr during negation application:\n' + str(expr))
    if isinstance(expr, UnExpr):
        if expr.op == UnOp.NOT:
            # we are trying to simplify !expr where expr == !expr.rhs
            # so we have !!expr.rhs
            # first, try simplifying !expr.rhs, and store the result in neg
            neg = apply_negation(expr.rhs)
            # if neg is of the form !neg.rhs, then we simplify !!neg.rhs to neg.rhs
            if isinstance(neg, UnExpr) and neg.op == NOT:
                return neg.rhs
            # otherwise, neg is some other non-negated expression that we can now attempt to negate like usual
            return apply_negation(neg)
        raise RuntimeError('Unexpected atom representing a UnExpr during negation application:\n' + str(expr))
    raise RuntimeError('Expression not recognised as an atom, BinExpr or UnExpr during negation application:\n' + str(expr))


def is_atom(expr):
    if isinstance(expr, str) or isinstance(expr, int):
        return True
    if isinstance(expr, BinExpr):
        return expr.op not in [BinOp.BICOND, BinOp.IMPL, BinOp.DISJ, BinOp.CONJ]
    if isinstance(expr, UnExpr):
        return expr.op not in [UnOp.NOT]
    raise RuntimeError('Unexpected expression type: ' + str(expr))


def to_disjunct_list(expr):
    def to_disjunct_list_helper(expr):
        if not isinstance(expr, BinExpr) or expr.op != BinOp.DISJ:
            return [expr]
        return to_disjunct_list_helper(expr.lhs) + to_disjunct_list_helper(expr.rhs)
    return to_disjunct_list_helper(to_dnf(expr))
