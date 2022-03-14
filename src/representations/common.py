__all__ = ["PreprocessingPipeline"]


class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add(self, step):
        self.steps.append(step)

    def run(self, arg):
        if not len(self.steps):
            return arg
        data = arg
        for step in self.steps:
            data = step(data)
        return data
