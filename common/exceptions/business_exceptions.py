

class BusinessException(Exception):
    
    def __init__(self, code: int = 1, msg: str = "业务异常"):
        super().__init__(msg)
        self.code = code
        self.msg = msg
        