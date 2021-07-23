class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        climbing_ways = [0 for _ in range(n)]
        climbing_ways[0] = 1
        climbing_ways[1] = 2
        for i in range(2, n):
            climbing_ways[i] = climbing_ways[i-1] + climbing_ways[i-2]
        return climbing_ways[-1]
        
    