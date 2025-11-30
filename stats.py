import time


class Stats:
    _timer = None
    performance_time = None
    state_lattice_joins = 0
    state_lattice_meets = 0
    iterations = 0

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
