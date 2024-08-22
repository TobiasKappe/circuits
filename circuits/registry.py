class ElementRegistry:
    IMPL_MAP = {}
    MOCK_MAP = {}

    @classmethod
    def add_impl(cls, name, mock_cls=None):
        def wrapper(impl):
            cls.IMPL_MAP[name] = impl
            if mock_cls is not None:
                cls.MOCK_MAP[impl] = mock_cls
            return impl

        return wrapper

    @classmethod
    def get_impl(cls, name):
        if name not in cls.IMPL_MAP:
            raise Exception(f'Unknown element: "{name}"')

        return cls.IMPL_MAP[name]

    @classmethod
    def get_mock(cls, elem_cls):
        if elem_cls not in cls.MOCK_MAP:
            raise Exception(f'Unknown mock: "{elem_cls.__name__}"')

        return cls.MOCK_MAP[elem_cls]
