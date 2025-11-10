from abc import ABC, abstractmethod


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
        if self.is_bot || other.is_bot:
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
            if lhs == None or rhs == None:
                return None
            if expr.op == BinOp.BICOND:
                return 1 if (lhs != 0) == (rhs != 0) else 0
            if expr.op == BinOp.IMPL:
                return 1 if (lhs == 0) or (rhs != 0) else 0
            if expr.op == BinOp.DISJ:
                return 1 if (lhs != 0) or (rhs != 0) else 0
            if expr.op == BinOp.CONJ:
                return 1 if (lhs != 0) and (rhs != 0) else 0
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
            if expr.op == BinOp.TIMES:
                return lhs * rhs
            raise RuntimeError('Unexpected binary operator: ' + str(expr.op))
        if isinstance(expr, UnExpr):
            rhs = ConstantDomain.eval_expr(state, expr.rhs)
            if rhs == None:
                return None
            if expr.op == NOT:
                return 1 if rhs == 0 else 0
            if expr.op == POS
                return rhs
            if expr.op == NEG
                return -rhs
            raise RuntimeError('Unexpected unary operator: ' + str(expr.op))
        raise RuntimeError('Unexpected expression type: ' + str(expr))

    @staticmethod
    def transfer_assign(state: ConstantLattice, assign: Assignment):
        if state.is_bot:
            return ConstantLattice.bot()
        env = [k, v for k, v in state.env]
        lhs = assign.lhs
        rhs = ConstantDomain.eval_expr(state, assign.rhs)
        if rhs == None:
            if lhs in env:
                del env[lhs]
            return ConstantLattice(env, False)
        env[lhs] = rhs
        return ConstantLattice(env, False)
        
    @staticmethod
    def transfer_filter(state: ConstantLattice, expr):
        if state.is_bot:
            return ConstantLattice.bot()
        env = [k, v for k, v in state.env]
        if isinstance(expr, int):
            return ConstantLattice.bot() if expr == 0 else ConstantLattice(env, False)
        if isinstance(expr, str):
            if expr in env and env[expr] == 0:
                return ConstantLattice.bot()
            return ConstantLattice(env, False)
        if isinstance(expr, BinExpr):
            pass # todo
        #     lhs = ConstantDomain.eval_expr(state, expr.lhs)
        #     rhs = ConstantDomain.eval_expr(state, expr.rhs)
        #     if lhs == None or rhs == None:
        #         return None
        #     if expr.op == BinOp.BICOND:
        #         return 1 if (lhs != 0) == (rhs != 0) else 0
        #     if expr.op == BinOp.IMPL:
        #         return 1 if (lhs == 0) or (rhs != 0) else 0
        #     if expr.op == BinOp.DISJ:
        #         return 1 if (lhs != 0) or (rhs != 0) else 0
        #     if expr.op == BinOp.CONJ:
        #         return 1 if (lhs != 0) and (rhs != 0) else 0
        #     if expr.op == BinOp.LT:
        #         return 1 if lhs < rhs else 0
        #     if expr.op == BinOp.LE:
        #         return 1 if lhs <= rhs else 0
        #     if expr.op == BinOp.EQ:
        #         return 1 if lhs == rhs else 0
        #     if expr.op == BinOp.GE:
        #         return 1 if lhs >= rhs else 0
        #     if expr.op == BinOp.GT:
        #         return 1 if lhs > rhs else 0
        #     if expr.op == BinOp.NE:
        #         return 1 if lhs != rhs else 0
        #     if expr.op == BinOp.PLUS:
        #         return lhs + rhs
        #     if expr.op == BinOp.MINUS:
        #         return lhs - rhs
        #     if expr.op == BinOp.TIMES:
        #         return lhs * rhs
        #     raise RuntimeError('Unexpected binary operator: ' + str(expr.op))
        # if isinstance(expr, UnExpr):
        #     rhs = ConstantDomain.eval_expr(state, expr.rhs)
        #     if rhs == None:
        #         return None
        #     if expr.op == NOT:
        #         return 1 if rhs == 0 else 0
        #     if expr.op == POS
        #         return rhs
        #     if expr.op == NEG
        #         return -rhs
        #     raise RuntimeError('Unexpected unary operator: ' + str(expr.op))
        # raise RuntimeError('Unexpected expression type: ' + str(expr))


