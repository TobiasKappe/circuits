from functools import singledispatchmethod

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Text')
class TextElement(Element):
    def is_resolvable(self):
        return False


@ElementRegistry.add_impl('Rectangle')
class RectangleElement(Element):
    def is_resolvable(self):
        return False


@ElementRegistry.add_impl('Arrow')
class ArrowElement(Element):
    def is_resolvable(self):
        return False


@ElementRegistry.add_impl('ImageAnnotation')
class ImageAnnotationElement(Element):
    def is_resolvable(self):
        return False


@ElementRegistry.add_impl('SevenSegDisplay')
class SevenSegDispleyElement(Element):
    def is_resolvable(self):
        return False


@ElementRegistry.add_impl('DigitalLed')
class DigitalLedElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, **kwargs):
        super().__init__(input, **kwargs)
        self.state = False
        self.inp1 = input

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        if self.inp1.value == []:
            return False

        return True

    def resolve(self):
        self.state = self.inp1.value[0]
        yield from ()
