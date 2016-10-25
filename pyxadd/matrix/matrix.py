class Matrix:

    def __init__(self, width, height, diagram):
        self._width = width
        self._height = height
        self._diagram = diagram

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def diagram(self):
        return self._diagram
