class IncrementalAverage:
    def __init__(self, total=0, size=0):
        self.total = total
        self.size = size

    def add(self, value):
        self.total += value
        self.size += 1

    def value(self):
        return self.total / self.size if self.size > 0 else 0


class ExponentialAverage:
    def __init__(self, init_value=0, step_size=0.1):
        self.curr_value = init_value
        self.step_size = step_size

    def add(self, value):
        self.curr_value += self.step_size * (value - self.curr_value)

    def value(self):
        return self.curr_value
