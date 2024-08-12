from functools import singledispatchmethod
from typing import List

from circuits.element import Element
from circuits.node import Node
from circuits.registry import ElementRegistry


class LogicElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        inp: List[Node],
        output1: Node,
        arity=2,
        bitwidth=1,
        **kwargs
    ):
        super().__init__(inp + [output1], **kwargs)
        self.inp = inp
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.arity = self.params[1]
        self.bitwidth = self.params[2]

    def is_resolvable(self):
        for i in range(self.bitwidth):
            if self.is_resolvable_per_bit(i):
                return True
        return False

    def is_resolvable_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
               i < len(inp.value) and \
               inp.value[i] is not None:
                return True
        return False

    def resolve(self):
        result = []
        for i in range(self.bitwidth):
            result.append(self.resolve_per_bit(i))

        self.output1.value = result
        yield self.output1

    def resolve_per_bit(self, i):
        raise NotImplementedError


@ElementRegistry.add_impl('AndGate')
class AndGateElement(LogicElement):
    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
               i < len(inp.value) and \
               inp.value[i] is False:
                return False
        return True


@ElementRegistry.add_impl('NorGate')
class NorGateElement(LogicElement):
    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
               i < len(inp.value) and \
               inp.value[i] is True:
                return False
        return True


@ElementRegistry.add_impl('OrGate')
class OrGateElement(LogicElement):
    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
               i < len(inp.value) and \
               inp.value[i] is True:
                return True
        return False


@ElementRegistry.add_impl('NandGate')
class NandGateElement(LogicElement):
    def resolve_per_bit(self, i):
        for inp in self.inp:
            if inp.value is not None and \
               i < len(inp.value) and \
               inp.value[i] is True:
                return False
        return True


class ParityGateElement(LogicElement):
    start_parity = None

    def resolve_per_bit(self, i):
        parity = self.start_parity
        for inp in self.inp:
            if inp.value is not None and \
               inp.value[i] is True:
                parity = not parity
        return parity


@ElementRegistry.add_impl('XorGate')
class XorGateElement(ParityGateElement):
    start_parity = False


@ElementRegistry.add_impl('XnorGate')
class XnorGateElement(ParityGateElement):
    start_parity = True


@ElementRegistry.add_impl('NotGate')
class NotGateElement(Element):
    @singledispatchmethod
    def __init__(self, inp1: Node, output1: Node, **kwargs):
        super().__init__([inp1, output1], **kwargs)
        self.inp1 = inp1
        self.output1 = output1

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)

    def is_resolvable(self):
        if self.inp1.value is None:
            return False
        for val in self.inp1.value:
            if val is None:
                return False
        return True

    def resolve(self):
        self.output1.value = [not v for v in self.inp1.value]
        yield self.output1
