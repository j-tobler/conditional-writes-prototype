from abc import ABC, abstractmethod
from itertools import combinations
from lark import Lark
from enum import Enum


class AnalysisMode(Enum):
    TRANSITIVE = 1
    NON_TRANSITIVE = 2

ANALYSIS_MODE = AnalysisMode.TRANSITIVE
PRECISION = -1

program_parser = Lark(
    r"""
    start:          procedure+
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
    def __init__(self, procedures: list):
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
    return Program([parse_procedure(proc) for proc in tree.children])

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


class Lattice(ABC):
    @staticmethod
    @abstractmethod
    def top():
        pass

    @staticmethod
    @abstractmethod
    def bot():
        pass

    @abstractmethod
    def is_bot(self):
        pass

    @abstractmethod
    def __or__(self, other):
        pass

    @abstractmethod
    def __and__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self <= other and self != other

    def __le__(self, other):
        return self & other == self

    def __ge__(self, other):
        return self | other == self

    def __gt__(self, other):
        return self >= other and self != other


class ConstantLattice(Lattice):
    def __init__(self, env, is_bot):
        self.env = env # read-only!
        self.is_bot = is_bot # read-only!

    @staticmethod
    def top():
        return ConstantLattice({}, False)

    @staticmethod
    def bot():
        return ConstantLattice({}, True)

    def is_bot(self):
        return self.is_bot

    def __or__(self, other):
        if self.is_bot:
            return other
        if other.is_bot:
            return self
        return ConstantLattice({k: v for k, v in self.env.items() if k in other.env and other.env[k] == v}, False)

    def __and__(self, other):
        if self.is_bot or other.is_bot:
            return bot()
        for k in set(self.env.keys()) & set(other.env.keys()):
            if self.env[k] != other.env[k]:
                return bot()
        return ConstantLattice(self.env | other.env, False)

    def __eq__(self, other):
        return self.env == other.env and self.is_bot == other.is_bot

    def __str__(self):
        if self.is_bot:
            return 'bot'
        if not self.env:
            return 'top'
        return ', '.join(v + ' -> ' + str(self.env[v]) for v in self.env)


class AbstractDomain(ABC):
    @staticmethod
    @abstractmethod
    def transfer_assign(state: Lattice, inst: Assignment) -> Lattice:
        pass
        
    @staticmethod
    @abstractmethod
    def transfer_filter(state: Lattice, inst: Lang) -> Lattice:
        pass

    @staticmethod
    @abstractmethod
    def havoc(state: Lattice, vars_to_remove) -> Lattice:
        pass

    @staticmethod
    @abstractmethod
    def constrained_vars(state: Lattice) -> set:
        pass

    @staticmethod
    @abstractmethod
    def top() -> Lattice:
        pass


class ConstantDomain(AbstractDomain):
    @staticmethod
    def eval_expr(state: ConstantLattice, expr) -> int | None:
        env = state.env
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            if expr in env:
                return env[expr]
            return None
        if isinstance(expr, BinExpr):
            lhs = ConstantDomain.eval_expr(state, expr.lhs)
            rhs = ConstantDomain.eval_expr(state, expr.rhs)
            if expr.op == BinOp.IMPL:
                if lhs == 0 or rhs != 0 and rhs != None:
                    return 1
                elif lhs != 0 and lhs != None and rhs == 0:
                    return 0
                return None
            if expr.op == BinOp.DISJ:
                if lhs != 0 and lhs != None or rhs != 0 and rhs != None:
                    return 1
                elif lhs == 0 and rhs == 0:
                    return 0
                return None
            if expr.op == BinOp.CONJ:
                if lhs != 0 and lhs != None and rhs != 0 and rhs != None:
                    return 1
                elif lhs == 0 or rhs == 0:
                    return 0
                return None
            if expr.op == BinOp.TIMES:
                if lhs == 0 or rhs == 0:
                    return 0
                if lhs == None or rhs == None:
                    return None
                return lhs * rhs
            if lhs == None or rhs == None:
                return None
            if expr.op == BinOp.BICOND:
                return 1 if (lhs == 0) == (rhs == 0) else 0
            if expr.op == BinOp.LT:
                return 1 if lhs < rhs else 0
            if expr.op == BinOp.LE:
                return 1 if lhs <= rhs else 0
            if expr.op == BinOp.EQ:
                return 1 if lhs == rhs else 0
            if expr.op == BinOp.GE:
                return 1 if lhs >= rhs else 0
            if expr.op == BinOp.GT:
                return 1 if lhs > rhs else 0
            if expr.op == BinOp.NE:
                return 1 if lhs != rhs else 0
            if expr.op == BinOp.PLUS:
                return lhs + rhs
            if expr.op == BinOp.MINUS:
                return lhs - rhs
            raise RuntimeError('Unexpected binary operator: ' + str(expr.op))
        if isinstance(expr, UnExpr):
            rhs = ConstantDomain.eval_expr(state, expr.rhs)
            if rhs == None:
                return None
            if expr.op == UnOp.NOT:
                return 1 if rhs == 0 else 0
            if expr.op == UnOp.POS:
                return rhs
            if expr.op == UnOp.NEG:
                return -rhs
            raise RuntimeError('Unexpected unary operator: ' + str(expr.op))
        raise RuntimeError('Unexpected expression type: ' + str(expr))

    @staticmethod
    def transfer_assign(state: ConstantLattice, assign: Assignment) -> ConstantLattice:
        if state.is_bot:
            return ConstantLattice.bot()
        env = state.env.copy()
        lhs = assign.lhs
        rhs = ConstantDomain.eval_expr(state, assign.rhs)
        if rhs == None:
            if lhs in env:
                del env[lhs]
            return ConstantLattice(env, False)
        env[lhs] = rhs
        return ConstantLattice(env, False)
        
    @staticmethod
    def transfer_filter(state: ConstantLattice, expr) -> ConstantLattice:
        """
        We only handle logical sentences over inequalities e1 <*> e2, where e1 or e2 is a variable and the other operand
        is an expression that evaluates to a constant in the current state.
        """
        if state.is_bot:
            return ConstantLattice.bot()
        val = ConstantDomain.eval_expr(state, expr)
        if val == 0:
            return ConstantLattice.bot() # condition is false
        elif val != None:
            return state # condition is true in all states
        if isinstance(expr, BinExpr):
            if expr.op == BinOp.DISJ:
                lhs_filter = ConstantDomain.transfer_filter(state, expr.lhs)
                rhs_filter = ConstantDomain.transfer_filter(state, expr.rhs)
                return lhs_filter | rhs_filter
            if expr.op == BinOp.CONJ:
                lhs_filter = ConstantDomain.transfer_filter(state, expr.lhs)
                rhs_filter = ConstantDomain.transfer_filter(state, expr.rhs)
                return lhs_filter & rhs_filter
            if expr.op == BinOp.EQ:
                lhs_eval = ConstantDomain.eval_expr(state, expr.lhs)
                rhs_eval = ConstantDomain.eval_expr(state, expr.rhs)
                if rhs_eval != None and isinstance(expr.lhs, str):
                    # ConstantDomain.eval(state, expr) == None we have lhs_eval == None and thus expr.lhs not in state
                    # map expr.lhs to expr.rhs
                    env = state.env.copy()
                    env[expr.lhs] = rhs_eval
                    return ConstantLattice(env, False)
                if lhs_eval != None and isinstance(expr.rhs, str):
                    env = state.env.copy()
                    env[expr.rhs] = lhs_eval
                    return ConstantLattice(env, False)
        # default: apply no filtering
        return state

    @staticmethod
    def havoc(state: ConstantLattice, vars_to_remove) -> ConstantLattice:
        if state.is_bot:
            return state
        env = {k: v for k, v in state.env.items() if k not in vars_to_remove}
        return ConstantLattice(env, False)

    @staticmethod
    def constrained_vars(state: ConstantLattice) -> set:
        return state.env.keys()

    @staticmethod
    def top() -> ConstantLattice:
        return ConstantLattice.top()
        

class InterferenceDomain(ABC):
    @staticmethod
    @abstractmethod
    def stabilise(state, interference):
        pass
        
    @staticmethod
    @abstractmethod
    def transitions(state, inst):
        pass

    @staticmethod
    @abstractmethod
    def close(interference):
        pass

    @staticmethod
    @abstractmethod
    def bot() -> Lattice:
        pass


class ConditionalWritesLattice(Lattice):
    """
    Maps each variable to the set of states under which it may be written to.
    Variables not in the map are implied to be mapped to bot.
    """
    def __init__(self, env):
        self.env = env # read-only!

    @staticmethod
    def top():
        # we never actually use this
        raise NotImplementedError('Method top is not implemented for the ConditionalWritesLattice.')

    @staticmethod
    def bot():
        return ConditionalWritesLattice({})

    def is_bot(self):
        return not self.env

    def __or__(self, other):
        env = self.env.copy()
        for k in other.env.keys():
            env[k] = env[k] | other.env[k] if k in env else other.env[k]
        return ConditionalWritesLattice(env)

    def __and__(self, other):
        env = self.env.copy()
        for k in other.env.keys():
            env[k] = env[k] & other.env[k] if k in env else other.env[k]
        return ConditionalWritesLattice(env).filter_out_bot()

    def filter_out_bot(self):
        self.env = {k: v for k, v in self.env if not v.is_bot()}
        return self

    def __eq__(self, other):
        return self.env == other.env


class ConditionalWritesDomain(InterferenceDomain):
    @staticmethod
    def stabilise(D, d: Lattice, i: ConditionalWritesLattice, N=-1) -> Lattice:
        # N == -1 specifies maximum precision.
        N = len(i.env.keys()) if N == -1 or N > len(i.env.keys()) else N
        X = ConditionalWritesDomain.stabilise_helper(D, set(), i, d, N, set())
        if N == len(i.env.keys()):
            return d | X
        VN = {frozenset(combo) for combo in combinations(i.env.keys(), N + 1)}
        Y = D.bot()
        for V in VN:
            meet = d
            for v in V:
                meet &= i.env(v)
            Y |= meet
        flattened = {v for subset in VN for v in subset}
        Y = D.havoc(Y, flattened)
        return d | X | Y

    @staticmethod
    def stabilise_helper(D, V, i, d, N, done) -> Lattice:
        if len(V) > N or V in done:
            return D.bot()
        ret = d
        for v in V:
            ret = ret & i.env[v]
        if ret.is_bot:
            done.add(frozenset(V))
            return ret
        ret = D.havoc(ret, V)
        for v in set(i.env.keys()) - V:
            ret |= ConditionalWritesDomain.stabilise_helper(D, V + {v}, i, d, N, done)
        done.add(frozenset(V))
        return ret

    @staticmethod
    def transitions(D, d: Lattice, assign: Assignment) -> ConditionalWritesLattice:
        return ConditionalWritesLattice({assign.lhs: d})

    @staticmethod
    def close(D, i: ConditionalWritesLattice):
        # if i.env[v] is bot, then close(i).env[v] will also be bot, so we don't need to consider unmapped vars
        while True:
            old_i = i.copy() # since lattices are read-only, this is safe
            # update i
            # only update mappings for variables not mapped to bot
            for v in i.env.keys():
                # update mapping for v
                # iterate through variables constrained in i.env[v] (we skip optimisation 2 for now)
                constrained = D.constrained_vars(i.env[v])
                i[v] = ConditionalWritesDomain.close_helper(D, set(), i, constrained, v, set())
            i |= old_i
            # check if reached fixpoint
            if i == old_i:
                return i

    @staticmethod
    def close_helper(D, V, i, powset_domain, v, done):
        if V in done:
            return D.bot()
        havoced = D.havoc(i.env[v], V)
        meet = D.top()
        for other_v in V:
            meet &= i.env[other_v]
        if meet <= havoced:
            done.add(frozenset(V))
            return meet
        ret = meet & havoced
        for other_v in powset_domain - V:
            ret |= ConditionalWritesDomain.close_helper(D, V + {other_v}, i, powset_domain, v, done)
        done.add(frozenset(V))
        return ret

    @staticmethod
    def bot() -> Lattice:
        return ConditionalWritesLattice.bot()
