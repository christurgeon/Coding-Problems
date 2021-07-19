class Solution:
    def isValid(self, s: str) -> bool:
        m = {
            "(" : ")",
            "[" : "]",
            "{" : "}"
        }
        q = deque()
        for c in s:
            if c in m:
                q.append(m[c])
            elif len(q) == 0:
                return False
            elif c != q.pop():
                return False
        return len(q) == 0