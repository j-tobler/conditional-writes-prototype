"""
=== Grammar ====================================================================

How to read this grammar:
- Single quotes ('') represent literals.
- A sequence (A B) is shorthand for (A ws* B).
- The modifier (+) means "one or more in a sequence".
- The modifier (*) means "zero or more in a sequence".
- The notation <A B> represents a sequence of A separated by B.
- We use [] to denote regex character classes.
- The non-terminal NEWLINE represents a standard newline character.

start       ::== procedure+
procedure   ::== 'proc' ws+ id '(' <id ','> ')' '{' statement* '}'
statement   ::== assignment | conditional | loop | call | assume
assignment  ::== id ':=' expr ';'
conditional ::== 'if' '(' expr ')' '{' statement* '}'
loop        ::== 'while' '(' expr ')' '{' statement* '}'
call        ::== id '(' <id ','> ')' ';'
assume      ::== 'assume' '(' ';'
expr        ::== value | uop value | value binop value
value       ::== id | num
uop         ::== '!' | '-'
binop       ::== '+' | '-' | '*' | '/' | '%' | '&&' | '||' | '==>' | '<===>'
id          ::== alpha alphanum*
alphanum    ::== alpha | num
alpha       ::== [A-Za-z]
num         ::== [0-9]
ws          ::== ' ' | NEWLINE
"""