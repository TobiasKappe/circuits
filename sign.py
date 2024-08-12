from functools import singledispatchmethod

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('SignExtend')
class SignExtendElement(Element):
    @singledispatchmethod
    def __init__(self, input: Node, output: Node, **kwargs):
        super().__init__([input, output], **kwargs)
        self.input = input
        self.output = output
        assert self.input.bitwidth == self.bitwidth

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]
        assert self.input.bitwidth == self.bitwidth

    def is_resolvable(self):
        return self.input.value is not None

    def resolve(self):
        self.output.value = \
            self.input.value + [self.input.value[-1]] * (32-self.bitwidth)
        yield self.output
