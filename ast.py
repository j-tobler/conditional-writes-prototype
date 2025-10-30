# === Top Level ================================================================

class Procedure:
    def __init__(self):
        pass

class Statement:
    def __init__(self):
        pass

class Expression:
    def __init__(self):
        pass


# === Statements ===============================================================

class Assignment(Statement):
    def __init__(self):
        pass

class Conditional(Statement):
    def __init__(self):
        pass

class Loop(Statement):
    def __init__(self):
        pass

class Call(Statement):
    def __init__(self):
        pass

class Assume(Statement):
    def __init__(self):
        pass

# === Expressions ==============================================================

class Value(Expression):
    def __init__(self):
        pass

class UnExpr(Expression):
    def __init__(self):
        pass

class BinExpr(Expression):
    def __init__(self):
        pass

# === Expressions > Values =====================================================

class Id(Value):
    def __init__(self):
        pass

class Literal(Value):
    def __init__(self):
        pass

# === Expressions > Unary Expressions ==========================================

class BoolNegation(UnExpr):
    def __init__(self):
        pass

class IntNegation(UnExpr):
    def __init__(self):
        pass

# === Expressions > Binary Expressions =========================================

class Addition(BinExpr):
    def __init__(self):
        pass

class Subtraction(BinExpr):
    def __init__(self):
        pass

class Multiplication(BinExpr):
    def __init__(self):
        pass

class Division(BinExpr):
    def __init__(self):
        pass

class Modulo(BinExpr):
    def __init__(self):
        pass

class Conjunction(BinExpr):
    def __init__(self):
        pass

class Disjunction(BinExpr):
    def __init__(self):
        pass

class Implication(BinExpr):
    def __init__(self):
        pass

class Biconditional(BinExpr):
    def __init__(self):
        pass
