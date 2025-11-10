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

r"""
start:          procedure+
procedure:      "proc" _SEP CNAME "{" _statement* "}"
_statement:     assignment | conditional | loop | assume
assignment:     CNAME ":=" expr ";"
conditional:    "if" _SEP expr "{" _statement* "}"
loop:           "while" _SEP expr "{" _statement* "}"
assume:         "assume" _SEP expr ";"
?expr:          implication | expr "<==>" implication
?implication:   disjunction | disjunction "==>" implication
?disjunction:   conjunction | disjunction "||" conjunction
?conjunction:   negation | conjunction "&&" negation
?negation:      atom | "!" negation
?atom:          inequality | "(" expr ")"
?inequality:    sum | sum (inequality_op sum)*
?sum:           term | sum sum_op term
?term:          signed_val | term term_op signed_val
?signed_val:    val | sum_op signed_val
val:            CNAME | INT | "(" sum ")"
id:             CNAME
num:            INT
!sum_op:        "+" | "-"
!term_op:       "*" | "/" | "%"
!inequality_op: "<" | "<=" | "==" | ">=" | ">" | "!="
_SEP:           WS+

%import common (WS, CNAME, INT)
%ignore WS
"""

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
    def eval_expr(state, expr):
        if 


    @staticmethod
    def transfer_assign(state: ConstantLattice, assignment: lark.tree.Tree):
        lhs: str = assignment.children[0].value
        rhs = assignment.children[1]
        if rhs.data == 'expr':
            pass
        elif rhs.data == 'uexpr':
            pass
        else:

        
    @staticmethod
    def transfer_filter(state: ConstantLattice, inst):
        pass
