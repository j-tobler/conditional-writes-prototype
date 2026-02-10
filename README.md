# Prototype for the Conditional-Writes Domain

## Installation

Navigate to this directory and create a Python virtual environment:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

You should now have `$(pwd)/venv/bin` in your PATH, containing the `pip` and `python` executables.

Install dependencies into the virtual environment:

`pip install lark==1.3.1`

See usage:

`python main.py --help`

Run a simple example:

`python main.py examples/spinlock.txt "disjunctive constants" -t`

Run the benchmarks from the paper (results are written to `results.txt`):

`python benchmarks.py`

NOTE: Some parts of our implementation involve iterating over Python's built-in `set` type, which uses a non-deterministic ordering. As a result, the behaviour of our implementation is non-deterministic in that it may give slightly different values for the number of state-lattice operations (for example, due to earlier exits from loops in some executions). However, verification success is still deterministic.