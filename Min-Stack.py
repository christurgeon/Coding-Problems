class MinStack(object):

    def __init__(self):
        # tuple (value, curr_min)
        self.__stack = []
        
    def push(self, x):
        minimum = self.getMin()
        self.__stack.append((x, min(minimum, x))) 

    def pop(self):
        self.__stack.pop()

    def top(self):
        if len(self.__stack) > 0:
            return self.__stack[-1][0]
        return None
        
    def getMin(self):
        if len(self.__stack) > 0:
            return self.__stack[-1][1]
        return 2**31 - 1 
