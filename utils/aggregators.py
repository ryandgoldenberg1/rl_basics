class IncrementalAverage:
    def __init__(self, total=0, size=0):
        self.total = total
        self.size = size

    def add(self, value):
        self.total += value
        self.size += 1

    def value(self):
        return self.total / self.size if self.size > 0 else 0
