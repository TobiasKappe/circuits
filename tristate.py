from functools import singledispatchmethod

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('TriState')
class TriStateElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp1: Node,
        output1: Node,
        state: Node,
        **kwargs
    ):
        self.inp1 = inp1
        self.output1 = output1
        self.state = state

        super().__init__([self.inp1, self.output1], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[1]

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        if None in self.inp1.value:
            return False
        if self.state.value is None:
            return False
        if self.state.value[0] is False:
            return False
        return True

    def resolve(self):
        self.output1.value = self.inp1.value
        yield self.output1
