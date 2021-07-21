class RecentCounter:

    def __init__(self):
        self.__tracker = deque()
        
    def ping(self, t: int) -> int:
        self.__tracker.appendleft(t)
        while abs(t - self.__tracker[-1]) > 3000:
            self.__tracker.pop()
        return len(self.__tracker)
        


# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)