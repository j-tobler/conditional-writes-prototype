import time


class Stats:
    _timer = None
    performance_time = None
    state_lattice_joins = 0
    state_lattice_meets = 0
    iterations = 0
    verified = False

    @staticmethod
    def to_str():
        return f'Time: {Stats.performance_time}\n'\
            f'State-Lattice Joins: {Stats.state_lattice_joins}\n'\
            f'State-Lattice Meets: {Stats.state_lattice_meets}\n'\
            f'State-Lattice Joins + Meets: {Stats.state_lattice_meets + Stats.state_lattice_joins}\n'\
            f'Iterations: {Stats.iterations}\n'\
            f'Verified: {Stats.verified}'

    @staticmethod
    def is_configured():
        return None not in [
            Stats.performance_time,
            Stats.state_lattice_joins,
            Stats.state_lattice_meets,
            Stats.iterations,
            Stats.verified
        ]

    @staticmethod
    def start_timer():
        Stats._timer = time.perf_counter()

    @staticmethod
    def end_timer():
        if Stats._timer == None:
            raise RuntimeError('end_timer called before start_timer.')
        delta = time.perf_counter() - Stats._timer
        Stats._timer = None
        Stats.performance_time = delta
