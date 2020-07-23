"""
    This is the base module of calculating the derivative of
    a customized norm

    you should only inherit this class and define:
        __init__()
        __call__()

    For example:
        def __call__(self, x):
            parameter1
            parameter2
            ...

        norm = cNorm(parameter1=0.1, parameter2=0.7)

    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""


class cNorm():
    r"""The base module of calculating the derivative of a customized norm

        Example:
            p_norm = cNorm(parameter1=0.1, parameter2=0.7)
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, x):
        r"""Calculate the derivative of your Norm

            For example:
                def __call__(self, x):
                    parameter1
                    parameter2
                    ...

        Args:
            x (tensor): input vector
        """
        return x

    def __repr__(self):
        _rep = []
        for key, value in self.__dict__.items():
            _rep.append(f"{key}={value}")

        _rep = ', '.join(_rep)

        return f"{type(self).__name__}({_rep})"
