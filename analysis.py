from abc import ABC, abstractmethod
from parser import *
from itertools import combinations


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
        return ConstantLattice({self.env | other.env}, False)

    def __eq__(self, other):
        return self.env == other.env and self.is_bot == other.is_bot


class AbstractDomain(ABC):
    @staticmethod
    @abstractmethod
    def transfer_assign(state: Lattice, inst: Lang) -> Lattice:
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

    @staticmethod
    def havoc(state: ConstantLattice, vars_to_remove) -> ConstantLattice:
        if state.is_bot:
            return state
        env = {k: v for k, v in state.env.items() if k not in vars_to_remove}
        return ConstantLattice(env, False)

    @staticmethod
    def constrained_vars(state: Lattice) -> set:
        return state.env.keys()
        

class InterferenceDomain(ABC):
    def __init__(self, D):
        self.D = D

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

    def __or__(self, other):
        env = self.env.copy()
        for k in other.env.keys:
            env[k] = env[k] | other.env[k] if k in env else other.env[k]
        return ConditionalWritesLattice(env)

    def __and__(self, other):
        env = self.env.copy()
        for k in other.env.keys:
            env[k] = env[k] & other.env[k] if k in env else other.env[k]
        return ConditionalWritesLattice(env).filter_out_bot()

    def filter_out_bot(self):
        self.env = {k: v for k, v in self.env if not v.is_bot()}
        return self

    def __eq__(self, other):
        return self.env == other.env


class ConditionalWritesDomain(InterferenceDomain):
    @staticmethod
    def stabilise(d: Lattice, i: ConditionalWritesLattice, N=-1) -> Lattice:
        # N == -1 specifies maximum precision.
        N = len(i.keys()) if N == -1 or N > len(i.keys()) else N
        X = stabilise_helper(set(), i, d, N, set())
        if N == len(i.keys()):
            return d | X
        VN = {frozenset(combo) for combo in combinations(i.keys(), N + 1)}
        Y = D.bot()
        for V in VN:
            meet = d
            for v in V:
                meet &= i.env(v)
            Y |= meet
        flattened = {v for subset in VN for v in subset}
        Y = D.havoc(Y, flattened)
        return d | X | Y

    def stabilise_helper(V, i, d, N, done) -> Lattice:
        if len(V) > N or V in done:
            return D.bot()
        ret = d
        for v in V:
            ret = ret & i.env[v]
        if ret.is_bot():
            done.add(V)
            return ret
        ret = D.havoc(ret, V)
        for v in set(i.keys()) - V:
            ret |= stabilise_helper(V + {v}, i, d, N, done)
        done.add(V)
        return ret

    @staticmethod
    def transitions(d: Lattice, assign: Assignment) -> ConditionalWritesLattice:
        return ConditionalWritesLattice({assign.lhs: d})

    @staticmethod
    def close(i: ConditionalWritesLattice):
        # if i.env[v] is bot, then close(i).env[v] will also be bot, so we don't need to consider unmapped vars
        while True:
            old_i = i.copy() # since lattices are read-only, this is safe
            # update i
            # only update mappings for variables not mapped to bot
            for v in i.keys():
                # update mapping for v
                # iterate through variables constrained in i.env[v]
                constrained = D.constrained_vars(i.env[v])
                i[v] = close_helper(set(), i, constrained, v, set())
            i |= old_i
            # check if reached fixpoint
            if i == old_i:
                return i

    def close_helper(V, i, powset_domain, v, done):
        if V in done:
            return D.bot()
        havoced = D.havoc(i.env[v], V)
        meet = D.top()
        for other_v in V:
            meet &= i.env[other_v]
        if meet <= havoced:
            done.add(V)
            return meet
        ret = meet & havoced
        for other_v in powset_domain - V:
            ret |= close_helper(V + {other_v}, i, powset_domain, v, done)
        done.add(V)
        return ret








        
