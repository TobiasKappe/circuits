from functools import singledispatchmethod

from riscv.data import RiscInteger

from ouca.circuits.element import Element
from ouca.circuits.node import Node
from ouca.circuits.registry import ElementRegistry


@ElementRegistry.add_impl('ProgramCounter')
class ProgramCounterElement(Element):
    @singledispatchmethod
    def __init__(
        self,
        dist: Node,
        jump: Node,
        reset: Node,
        ctr: Node,
        clock: Node,
        **kwargs
    ):
        super().__init__([dist, jump, reset, ctr, clock], **kwargs)
        self.dist = dist
        self.jump = jump
        self.reset = reset
        self.ctr = ctr
        self.clock = clock
        self.prev_clock = None
        self.value = RiscInteger(0)
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = RiscInteger(0)
        self.prev_clock = None
        self.validate()

    def validate(self):
        assert self.dist.bitwidth == 32
        assert self.jump.bitwidth == 1
        assert self.reset.bitwidth == 1
        assert self.clock.bitwidth == 1
        assert self.ctr.bitwidth == 32

    def is_resolvable(self):
        return True

    def resolve(self):
        if self.reset.value is not None and self.reset.value[0] is True:
            self.value = RiscInteger(0)
        elif (self.clock.value is not None and self.clock.value[0] is True and
              self.prev_clock is False):
            if self.jump.value is not None and self.jump.value[0] is True:
                if self.dist.value is not None:
                    self.value += RiscInteger(self.dist.value)
            else:
                self.value += RiscInteger(4)

        self.ctr.value = self.value.bits
        yield self.ctr

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]
