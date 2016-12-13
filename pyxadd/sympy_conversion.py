import sympy


class SympyConverter(object):
    def convert(self, expression):
        if isinstance(expression, sympy.Add):
            return self.add_batch([self.convert(arg) for arg in expression.args])
        elif isinstance(expression, sympy.Mul):
            return self.times_batch([self.convert(arg) for arg in expression.args])
        elif isinstance(expression, sympy.Symbol):
            return self.symbol(str(expression))

        try:
            expression = float(expression)
            return self.int(int(expression)) if expression.is_integer() else self.float(expression)
        except ValueError:
            return self.custom(expression)

    def add_batch(self, values):
        result = values[0]
        for i in range(1, len(values)):
            result = self.add(result, values[i])
        return result

    def add(self, val1, val2):
        raise NotImplementedError()

    def times_batch(self, values):
        result = values[0]
        for i in range(1, len(values)):
            result = self.times(result, values[i])
        return result

    def times(self, val1, val2):
        raise NotImplementedError()

    def symbol(self, value):
        raise NotImplementedError()

    def int(self, value):
        raise NotImplementedError()

    def float(self, value):
        raise NotImplementedError()

    def custom(self, value):
        raise NotImplementedError()
