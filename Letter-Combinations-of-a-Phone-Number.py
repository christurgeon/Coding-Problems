class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        digits_map = {
            "2":  ["a", "b", "c"],
            "3":  ["d", "e", "f"],
            "4":  ["g", "h", "i"],
            "5":  ["j", "k", "l"],
            "6":  ["m", "n", "o"],
            "7":  ["p", "q", "r", "s"],
            "8":  ["t", "u", "v"],
            "9":  ["w", "x", "y", "z"]    
        }
        agg = []
        if len(digits) > 0:
            self.helper(0, digits, digits_map, agg, "") # populate agg using reference
            return agg
        else:
            return agg
        
    def helper(self, idx, digits, digits_map, aggregator, curr):
        if idx == len(digits):
            aggregator.append(curr)
        else:
            for c in digits_map[digits[idx]]:
                self.helper(idx+1, digits, digits_map, aggregator, curr+c)
       
        