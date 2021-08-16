class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        digits[-1] = 1 if len(digits) == 0 else digits[-1] + 1
        i = len(digits) - 1
        while i > 0 and digits[i] == 10:
            digits[i] = 0
            digits[i-1] = digits[i-1] + 1
            i -= 1
        if digits[0] == 10:
            digits[0] = 1
            digits.append(0)
        return digits
        
