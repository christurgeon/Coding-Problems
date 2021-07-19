class Solution:
    
    def helper(self, n_open: int, n_close: int, agg: str, result: List[int]):
        if n_open == 0 and n_close == 0:
            result.append(agg)
            return
        
        # if open ( is available then add one and recurse
        if n_open > 0:
            self.helper(n_open-1, n_close, agg + "(", result)
            
        # if there are extra ) available, use one
        if n_close > n_open:
            self.helper(n_open, n_close-1, agg + ")", result)
    
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        self.helper(n, n, "", result)
        return result