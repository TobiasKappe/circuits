from circuits.element import Element
from circuits.node import Node


class MockElement(Element):
    inputs = None
    outputs = None

    @property
    def fields(self):
        return self.inputs | self.outputs

    def __init__(self, **kwargs):
        nodes = []
        for field, bitwidth in self.fields.items():
            node = Node(bitwidth=bitwidth)
            setattr(self, field, node)
            nodes.append(node)

        for field, bitwidth in self.outputs.items():
            if field in kwargs:
                getattr(self, field).value = kwargs[field]
            else:
                getattr(self, field).value = [False] * bitwidth

        super().__init__(nodes, **kwargs)

    def is_resolvable(self):
        return True

    def resolve(self):
        for field in self.outputs:
            yield getattr(self, field)
