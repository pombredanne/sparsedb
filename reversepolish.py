class ReversePolish:
    def __init__(self, tokeniser, dispatcher, unwrapper):
        self._tok = tokeniser
        self._dis = dispatcher
        self._unwrap = unwrapper

    def execute(self, expr):
        stack = []
        tokens = self._tok(expr)
        for val,op,arity in tokens:
            #print('%s -> ' % stack, end='')
            if op:
                stack, operands = ReversePolish._pop(stack, arity)
                stack.append(self._dis[val](*(self._unwrap(o) for o in operands)))
            else:
                stack.append(val)
            #print(stack)

        return self._unwrap(stack.pop())

    @staticmethod
    def _pop(lst, n):
        if len(lst) < n:
            raise ValueError("not enough elements in stack")
        return lst[:-n],lst[-n:]


simple_tokeniser_ops = {
        'bool': {
            '&': 2,     # and
            '|': 2,     # or
            '^': 2,     # xor
            '-': 2,     # diff
            '!': 1      # not
        }
}
def simple_tokeniser(typ,expr):
    ops = simple_tokeniser_ops[typ]

    return [
        (v, v in set(ops), ops.get(v, 0))
        for v in expr.split(' ')
    ]
