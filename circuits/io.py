from functools import singledispatchmethod

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('Input')
class InputElement(Element):
    @singledispatchmethod
    def __init__(self, node: Node, state=None, **kwargs):
        super().__init__([node], **kwargs)
        self.output1 = node
        self.state = state

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.state = []
        self.bitwidth = self.params[1]
        raw_state = raw_element['customData']['values']['state']
        for i in range(self.bitwidth):
            self.state.append(raw_state & (1 << i) > 0)

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = self.state
        yield self.output1


@ElementRegistry.add_impl('Output')
class OutputElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, value=None, **kwargs):
        super().__init__([inp1], **kwargs)
        self.inp1 = inp1
        self.value = value

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = None

    def is_resolvable(self):
        return True

    def resolve(self):
        self.value = self.inp1.value
        yield from ()


@ElementRegistry.add_impl('Button')
class ButtonElement(InputElement):
    pass
