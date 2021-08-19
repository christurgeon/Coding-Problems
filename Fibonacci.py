# classic recursive
def fib(N):
	if N == 0: 
        return 0
	if N == 1: 
        return 1
	return fib(N-1) + fib(N-2)

# memoized recursive
memo = {}
def fib(N):
	if N == 0: 
        return 0
	if N == 1: 
        return 1

	if N-1 not in memo: 
        memo[N-1] = fib(N-1)
	if N-2 not in memo: 
        memo[N-2] = fib(N-2)
	return memo[N-1] + memo[N-2]

# iterative
class Solution:
    def fib(self, n: int) -> int:
        if n in [0]:
            return 0
        if n in [1, 2]:
            return 1
        fst, scd = 0, 1
        for _ in range(n-1):
            fst, scd = scd, fst + scd
        return scd
