from abc import ABC, abstractmethod
from parser import *


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
        self.env = env
        self.is_bot = is_bot

    @staticmethod
    def top():
        return ConstantLattice({}, False)

    @staticmethod
    def bot():
        return ConstantLattice({}, True)

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
        return ConstantLattice({self.env | other.env}, False)

    def __eq__(self, other):
        return self.env == other.env and self.is_bot == other.is_bot


class AbstractDomain(ABC):
    @staticmethod
    @abstractmethod
    def transfer_assign(state, inst):
        pass
        
    @staticmethod
    @abstractmethod
    def transfer_filter(state, inst):
        pass


class ConstantDomain(AbstractDomain):
    @staticmethod
    def eval_expr(state: ConstantLattice, expr) -> int | None:
        env = state.env
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            if expr in env:
                return env[env]
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
            if expr.op == NOT:
                return 1 if rhs == 0 else 0
            if expr.op == POS:
                return rhs
            if expr.op == NEG:
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
