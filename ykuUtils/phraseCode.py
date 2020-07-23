"""
    This module will simplify a python source code
    it will get rid of comment lines and redundant indentation
    Copyright 2019 Yang Kaiyu https://github.com/yku12cn
"""
from io import StringIO, IOBase
import token
import tokenize


def _stripCom(source):
    r"""strip comments from code by token method
    credit: https://stackoverflow.com/a/1769577/10985639
    original author: Ned Batchelder
    """
    if not isinstance(source, IOBase):
        raise TypeError("Input shall be an IOBase object")

    mod = StringIO()

    prev_toktype = token.INDENT
    last_lineno = -1
    last_col = 0

    tokgen = tokenize.generate_tokens(source.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), _ in tokgen:
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            mod.write(" " * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            pass
        elif toktype == tokenize.COMMENT:
            pass
        else:
            mod.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno
    return mod.getvalue()


def stripCode(code):
    r"""strip comments from code

    Args:
        code (str/filelike obj): code to be processed

    Returns:
        str: stripped code
    """
    if isinstance(code, str):
        code = StringIO(code)
    code = _stripCom(code)
    result = []
    for line in iter(code.splitlines()):
        if line.isspace() or not line:  # Jump over empty lines
            continue
        result.append(line.rstrip())
    result.append("")
    return "\n".join(result)


if __name__ == '__main__':
    # Demo
    democode = '''\

  # this is a demo


    def NNstruct(self):
        r\"\"\"woiwjife
        Arguments:
            saveModel(filename, [mkdirF=True]):
            filename = "path to model"
        \"\"\"
        a = \'\'\'woiwjife
Arguments:
wwwwel"
\'\'\'
        self.fc_layer = nn.Sequential(\t

            # nn.Sigmoid(),

            nn.Linear(784, 10)  # aiwiwi
            # nn.ReLU(inplace=True)
        )


'''
    print(stripCode(democode))
