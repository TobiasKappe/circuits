from functools import singledispatchmethod

from circuits.element import Element
from circuits.mock import MockElement
from circuits.node import Node
from circuits.registry import ElementRegistry
from circuits.utils import array_to_int, int_to_array, add_fixed_width


class MockProgramCounterElement(MockElement):
    inputs = {
        'dist': 32,
        'jump': 1,
        'reset': 1,
        'clock': 1,
    }

    outputs = {
        'ctr': 32,
    }


@ElementRegistry.add_impl(
    'ProgramCounter',
    MockProgramCounterElement
)
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
        self.value = 0
        self.validate()

    @__init__.register
    def load(self, raw_element: dict, **kwargs):
        super().__init__(raw_element, **kwargs)
        self.value = 0
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
            self.value = 0
        elif (self.clock.value is not None and self.clock.value[0] is True and
              self.prev_clock is False):
            if self.jump.value is not None and self.jump.value[0] is True:
                if self.dist.value is not None:
                    self.value = add_fixed_width(
                        self.value,
                        array_to_int(self.dist.value)
                    )
            else:
                self.value = add_fixed_width(self.value, 4)

        self.ctr.value = int_to_array(self.value)
        yield self.ctr

        if self.clock.value is not None:
            self.prev_clock = self.clock.value[0]
