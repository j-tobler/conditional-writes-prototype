from abc import ABC, abstractmethod
from itertools import combinations
from enum import Enum
from parsing import Assignment, Lang

class GlobalCounter:
    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

    def __str__(self):
        return str(self.count)

CONSTANT_LATTICE_CALLS = GlobalCounter()
DISJ_CONSTANT_LATTICE_CALLS = GlobalCounter()


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
    def constrained_vars(self):
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

    @abstractmethod
    def __hash__(self):
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
    def __init__(self, env: dict, is_bot: bool):
        self._env = env
        self._is_bot = is_bot
        self._hash_cache = None

    @staticmethod
    def top():
        return ConstantLattice({}, False)

    @staticmethod
    def bot():
        return ConstantLattice({}, True)

    def is_bot(self):
        return self._is_bot

    def constrained_vars(self):
        return self._env.keys()

    def __or__(self, other):
        global CONSTANT_LATTICE_CALLS
        CONSTANT_LATTICE_CALLS.inc()
        
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        return ConstantLattice({k: v for k, v in self._env.items() if k in other._env and other._env[k] == v}, False)

    def __and__(self, other):
        global CONSTANT_LATTICE_CALLS
        CONSTANT_LATTICE_CALLS.inc()
        
        if self.is_bot() or other.is_bot():
            return ConstantLattice.bot()
        for k in self._env.keys() & other._env.keys():
            if self._env[k] != other._env[k]:
                return ConstantLattice.bot()
        return ConstantLattice(self._env | other._env, False)

    def __eq__(self, other):
        return self._env == other._env and self.is_bot() == other.is_bot()

    def __hash__(self):
        if self._hash_cache != None:
            return self._hash_cache
        self._hash_cache = hash(frozenset(self._env.items())) + hash(self._is_bot)
        return self._hash_cache

    def __str__(self):
        if self.is_bot():
            return 'bot'
        if not self._env:
            return 'top'
        return ', '.join(v + ' -> ' + str(self._env[v]) for v in self._env)


class ConstantDisjunctionLattice(Lattice):
    def __init__(self, env: set):
        self._env = env
        self._hash_cache = None

    @staticmethod
    def top():
        return ConstantDisjunctionLattice({ConstantLattice.top()})

    @staticmethod
    def bot():
        return ConstantDisjunctionLattice(set())

    def is_bot(self):
        return not self._env

    def constrained_vars(self):
        ret = set()
        for x in self._env:
            ret |= x.constrained_vars()
        return ret

    def well_formed(self):
        maximal_env = {x for x in self._env if not any(x < y for y in self._env)}
        if maximal_env == {ConstantLattice.bot()}:
            # if maximal_env contains bot then maximal_env == {bot} since it's a maximal set
            return ConstantDisjunctionLattice.bot()
        return ConstantDisjunctionLattice(maximal_env)

    def apply_to_each_disjunct(self, func):
        return ConstantDisjunctionLattice({func(x) for x in self._env}).well_formed()

    def __or__(self, other):
        global DISJ_CONSTANT_LATTICE_CALLS
        DISJ_CONSTANT_LATTICE_CALLS.inc()
        return ConstantDisjunctionLattice(self._env | other._env).well_formed()

    def __and__(self, other):
        global DISJ_CONSTANT_LATTICE_CALLS
        DISJ_CONSTANT_LATTICE_CALLS.inc()
        return ConstantDisjunctionLattice({x & y for x in self._env for y in other._env}).well_formed()

    def __eq__(self, other):
        return self._env == other._env

    def __hash__(self):
        if self._hash_cache != None:
            return self._hash_cache
        self._hash_cache = hash(frozenset(self._env))
        return self._hash_cache

    def __str__(self):
        if not self._env:
            return 'bot'
        return ' \\/ '.join(str(x) for x in self._env)


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

    @staticmethod
    @abstractmethod
    def bot() -> Lattice:
        pass


class ConstantDomain(AbstractDomain):
    @staticmethod
    def eval_expr(state: ConstantLattice, expr) -> int | None:
        env = state._env
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
        if state.is_bot():
            return ConstantLattice.bot()
        env = state._env.copy()
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
        if state.is_bot():
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
                    env = state._env.copy()
                    env[expr.lhs] = rhs_eval
                    return ConstantLattice(env, False)
                if lhs_eval != None and isinstance(expr.rhs, str):
                    env = state._env.copy()
                    env[expr.rhs] = lhs_eval
                    return ConstantLattice(env, False)
        # default: apply no filtering
        return state

    @staticmethod
    def havoc(state: ConstantLattice, vars_to_remove) -> ConstantLattice:
        if state.is_bot():
            return state
        env = {k: v for k, v in state._env.items() if k not in vars_to_remove}
        return ConstantLattice(env, False)

    @staticmethod
    def constrained_vars(state: ConstantLattice) -> set:
        return state.constrained_vars()

    @staticmethod
    def top() -> ConstantLattice:
        return ConstantLattice.top()

    @staticmethod
    def bot() -> ConstantLattice:
        return ConstantLattice.bot()
        

class DisjunctiveConstantsDomain(AbstractDomain):
    @staticmethod
    def transfer_assign(state: ConstantDisjunctionLattice, assign: Assignment) -> ConstantDisjunctionLattice:
        return state.apply_to_each_disjunct(lambda x: ConstantDomain.transfer_assign(x, assign))

    @staticmethod
    def transfer_filter(state: ConstantDisjunctionLattice, expr) -> ConstantDisjunctionLattice:
        return state.apply_to_each_disjunct(lambda x: ConstantDomain.transfer_filter(x, expr))

    @staticmethod
    def havoc(state: ConstantDisjunctionLattice, vars_to_remove) -> ConstantDisjunctionLattice:
        return state.apply_to_each_disjunct(lambda x: ConstantDomain.havoc(x, vars_to_remove))

    @staticmethod
    def constrained_vars(state: ConstantDisjunctionLattice) -> set:
        return state.constrained_vars()

    @staticmethod
    def top() -> ConstantDisjunctionLattice:
        return ConstantDisjunctionLattice.top()

    @staticmethod
    def bot() -> ConstantDisjunctionLattice:
        return ConstantDisjunctionLattice.bot()


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
        self._env = env

    @staticmethod
    def top():
        # we never actually use this
        raise NotImplementedError('Method top is not implemented for the ConditionalWritesLattice.')

    @staticmethod
    def bot():
        return ConditionalWritesLattice({})

    def is_bot(self):
        return not self._env

    def constrained_vars(self):
        raise NotImplementedError('Function constrained_vars not implemented for ConditionalWritesLattice.')

    def __or__(self, other):
        env = self._env.copy()
        for k in other._env.keys():
            env[k] = env[k] | other._env[k] if k in env else other._env[k]
        return ConditionalWritesLattice(env)

    def __and__(self, other):
        env = self._env.copy()
        for k in other._env.keys():
            env[k] = env[k] & other._env[k] if k in env else other._env[k]
        return ConditionalWritesLattice(env).filter_out_bot()

    def filter_out_bot(self):
        self._env = {k: v for k, v in self._env if not v.is_bot()()}
        return self

    def __eq__(self, other):
        return self._env == other._env

    def copy(self):
        return ConditionalWritesLattice

    def __str__(self):
        return '\n'.join(v + ' -> [' + str(d) + ']' for v, d in self._env.items())


class ConditionalWritesDomain(InterferenceDomain):
    @staticmethod
    def stabilise(D, d: Lattice, i: ConditionalWritesLattice, N=-1) -> Lattice:
        # N == -1 specifies maximum precision.
        N = len(i._env.keys()) if N == -1 or N > len(i._env.keys()) else N
        X = ConditionalWritesDomain.stabilise_helper(D, set(), i, d, N, set())
        if N == len(i._env.keys()):
            return d | X
        VN = {frozenset(combo) for combo in combinations(i._env.keys(), N + 1)}
        Y = D.bot()
        for V in VN:
            meet = d
            for v in V:
                meet &= i._env(v)
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
            ret = ret & i._env[v]
        if ret.is_bot():
            done.add(frozenset(V))
            return ret
        ret = D.havoc(ret, V)
        for v in set(i._env.keys()) - V:
            ret |= ConditionalWritesDomain.stabilise_helper(D, V | {v}, i, d, N, done)
        done.add(frozenset(V))
        return ret

    @staticmethod
    def transitions(D, d: Lattice, assign: Assignment) -> ConditionalWritesLattice:
        return ConditionalWritesLattice({assign.lhs: d})

    @staticmethod
    def close(D, i: ConditionalWritesLattice):
        # if i._env[v] is bot, then close(i)._env[v] will also be bot, so we don't need to consider unmapped vars
        while True:
            old_i = i
            i = ConditionalWritesLattice(i._env.copy())
            # update i
            # only update mappings for variables not mapped to bot
            for v in i._env.keys():
                # update mapping for v
                # iterate through variables constrained in i._env[v] (we skip optimisation 2 for now)
                constrained = D.constrained_vars(i._env[v])
                i._env[v] = ConditionalWritesDomain.close_helper(D, set(), i, constrained, v, set())
            # i |= old_i
            # check if reached fixpoint
            if i == old_i:
                return i

    @staticmethod
    def close_helper(D, V, i, powset_domain, v, done):
        if V in done:
            return D.bot()
        havoced = D.havoc(i._env[v], V)
        meet = D.top()
        for other_v in V:
            if other_v in i._env:
                meet &= i._env[other_v]
            else:
                meet = D.bot()
                break
        if meet <= havoced:
            done.add(frozenset(V))
            return meet
        ret = meet & havoced
        for other_v in powset_domain - V:
            ret |= ConditionalWritesDomain.close_helper(D, V | {other_v}, i, powset_domain, v, done)
        done.add(frozenset(V))
        return ret

    @staticmethod
    def bot() -> Lattice:
        return ConditionalWritesLattice.bot()
