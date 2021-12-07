class InputItem:
    def __init__(self, content):
        self.content = content
        self.process_content = []

    def apply(self, funcs):
        if self.process_content:
            return self.process_content
        for k, v in self.content.items():
            for func in funcs[k]:
                v = func(v)
            self.process_content.append(v)
        self.content = None
        return self.process_content
