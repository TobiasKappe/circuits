from functools import singledispatchmethod
from typing import List

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


@ElementRegistry.add_impl('ConstantVal')
class ConstantValElement(Element):
    @singledispatchmethod
    def __init__(self, value: List[bool], output1: Node, **kwargs):
        super().__init__([output1], **kwargs)
        self.value = value
        self.output1 = output1
        assert self.bitwidth == len(value)

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = [v == "1" for v in self.params[2]][::-1]
        self.bitwidth = self.params[1]
        assert self.bitwidth == len(self.value)

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = self.value
        yield self.output1


class FixedValElement(Element):
    fixed_value = None

    @singledispatchmethod
    def __init__(self, output1: Node, **kwargs):
        super().__init__([output1], **kwargs)
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.bitwidth = self.params[0]

    def is_resolvable(self):
        return True

    def resolve(self):
        self.output1.value = [self.fixed_value] * self.bitwidth
        yield self.output1


@ElementRegistry.add_impl('Ground')
class GroundElement(FixedValElement):
    fixed_value = False


@ElementRegistry.add_impl('Power')
class PowerElement(FixedValElement):
    fixed_value = True
