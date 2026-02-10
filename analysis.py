# Copyright 2025 The Commonwealth of Australia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from itertools import combinations
from cfg import *
from stats import Stats
from config import Config
from typing import TypeVar, Generic


class Lattice(ABC):
    @staticmethod
    @abstractmethod
    def top() -> Lattice:
        pass

    @staticmethod
    @abstractmethod
    def bot() -> Lattice:
        pass

    @abstractmethod
    def is_bot(self) -> bool:
        pass

    @abstractmethod
    def constrained_vars(self) -> set:
        pass

    @abstractmethod
    def implies_expr(self, expr) -> bool:
        pass

    @abstractmethod
    def __or__(self, other) -> Lattice:
        pass

    @abstractmethod
    def __and__(self, other) -> Lattice:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
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


L = TypeVar('L', bound=Lattice)


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
        return set(self._env.keys())

    def implies_expr(self, expr):
        # this is a very conservative implementation, designed to handle just the examples for the paper
        # we only handle expressions of the form '<var> == <int>'
        if self.is_bot():
            return True
        if not isinstance(expr, BinExpr) or expr.op != BinOp.EQ or not isinstance(expr.lhs, str) or \
            not isinstance(expr.rhs, int):
            raise RuntimeError('Assertion too complex to verify.')
        return expr.lhs in self._env and self._env[expr.lhs] == expr.rhs

    def __or__(self, other):
        Stats.state_lattice_joins += 1
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        return ConstantLattice({k: v for k, v in self._env.items() if k in other._env and other._env[k] == v}, False)

    def __and__(self, other):
        Stats.state_lattice_meets += 1
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

    def implies_expr(self, expr):
        return self.is_bot() or all(x.implies_expr(expr) for x in self._env)

    def __or__(self, other):
        Stats.state_lattice_joins += 1
        return ConstantDisjunctionLattice(self._env | other._env).well_formed()

    def __and__(self, other):
        Stats.state_lattice_meets += 1
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
        if len(self._env) == 1 and next(iter(self._env)) == ConstantLattice.top():
            return 'top'
        return ' \\/ '.join(f'[{x}]' for x in self._env)


class AbstractDomain(ABC, Generic[L]):
    @staticmethod
    @abstractmethod
    def transfer_assign(state: L, assign: Assignment) -> L:
        pass
        
    @staticmethod
    @abstractmethod
    def transfer_filter(state: L, expr: Lang) -> L:
        pass

    @staticmethod
    @abstractmethod
    def havoc(state: L, vars_to_remove) -> L:
        pass

    @staticmethod
    @abstractmethod
    def constrained_vars(state: L) -> set:
        pass

    @staticmethod
    @abstractmethod
    def top() -> L:
        pass

    @staticmethod
    @abstractmethod
    def bot() -> L:
        pass


class ConstantDomain(AbstractDomain[ConstantLattice]):
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
        for i in range(len(assign.lhs)):
            lhs = assign.lhs[i]
            rhs = ConstantDomain.eval_expr(state, assign.rhs[i])
            if rhs == None:
                if lhs in env:
                    del env[lhs]
            else:
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
        return set(state.constrained_vars())

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
        """
        For each disjunct, we apply the filter just as we would in the constant domain.
        The exception is filters that are themselves disjunctions. For example, suppose we have:
        x || y
        and we would like to apply the filter:
        a || b.
        By default, we would do:
        x && (a || b) || y && (a || b)
        But we would actually prefer:
        x && a || x && b || y && a || y && b.
        That is, we split the disjunct x by applying filters a and b separately, rather than as one expression.
        """
        if state.is_bot():
            return ConstantDisjunctionLattice.bot()
        disjuncts = to_disjunct_list(expr)
        # filter by each disjunct separately and then join the results
        new_state = DisjunctiveConstantsDomain.bot()
        for d in disjuncts:
            new_state |= state.apply_to_each_disjunct(lambda x: ConstantDomain.transfer_filter(x, d))
        return new_state

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
    def stabilise(D, state, interference) -> Lattice:
        pass

    @staticmethod
    def capture_interference(I, D, state, interference):
        new_state = I.stabilise(D, state, interference)
        if not Config.transitive_mode:
            while new_state != state:
                state = new_state
                new_state = I.stabilise(D, state, interference)
        return new_state
        
    @staticmethod
    @abstractmethod
    def transitions(D, state, assign) -> Lattice:
        pass

    @staticmethod
    @abstractmethod
    def close(D, interference) -> Lattice:
        pass

    @staticmethod
    @abstractmethod
    def bot(D) -> Lattice:
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

    def implies_expr(self, expr):
        raise RuntimeError('implies_expr called on an interference lattice.')

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
        self._env = {k: v for k, v in self._env.items() if not v.is_bot()}
        return self

    def __eq__(self, other):
        return self._env == other._env

    def copy(self):
        return ConditionalWritesLattice

    def __str__(self):
        if not self._env:
            return 'bot'
        return '\n'.join(v + ' -> [' + str(d) + ']' for v, d in self._env.items())


class ConditionalWritesDomain(InterferenceDomain):
    @staticmethod
    def stabilise(D, state: Lattice, interference: ConditionalWritesLattice) -> Lattice:
        N = Config.precision
        # N == -1 specifies maximum precision.
        N = len(interference._env.keys()) if N == -1 or N > len(interference._env.keys()) else N
        X = ConditionalWritesDomain.stabilise_helper(D, set(), interference, state, N, set())
        if N == len(interference._env.keys()):
            return state | X
        VN = {frozenset(combo) for combo in combinations(interference._env.keys(), N + 1)}
        Y = D.bot()
        for V in VN:
            meet = state
            for v in V:
                meet &= interference._env[v]
            Y |= meet
        flattened = {v for subset in VN for v in subset}
        Y = D.havoc(Y, flattened)
        return state | X | Y

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
    def transitions(D, state: Lattice, assign: Assignment) -> ConditionalWritesLattice:
        return ConditionalWritesLattice({lhs: state for lhs in assign.lhs})

    @staticmethod
    def close(D, interference: ConditionalWritesLattice):
        # if interference._env[v] is bot, then close(interference)._env[v] will also be bot, so we don't need to consider unmapped vars
        while True:
            old_i = interference
            interference = ConditionalWritesLattice(interference._env.copy())
            # update interference
            # only update mappings for variables not mapped to bot
            for v in interference._env.keys():
                # update mapping for v
                # iterate through variables constrained in interference._env[v]
                constrained = D.constrained_vars(interference._env[v])
                interference._env[v] = ConditionalWritesDomain.close_helper(D, set(), interference, constrained, v, set())
            # interference |= old_i
            # check if reached fixpoint
            if interference == old_i:
                return interference

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
    def bot(D) -> Lattice:
        return ConditionalWritesLattice.bot()
