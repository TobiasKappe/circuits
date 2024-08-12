from functools import singledispatchmethod

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Clock')
class ClockElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        output1: Node,
        **kwargs
    ):
        self.output1 = output1
        self.state = [False]

        super().__init__([self.output1], **kwargs)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.state = [False]

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = self.state
        yield self.output1
