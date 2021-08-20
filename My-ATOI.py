class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if len(s) == 0:
            return 0
        
        i = 0
        sign = None
        if s[0] in ["+", "-"]:
            sign = s[0]
            i += 1
        MAX_INT, MIN_INT = 2147483647, -2147483648
        
        new_s = []
        for i in range(i, len(s)):
            if s[i].isdigit():
                new_s.append(s[i])
            else:
                break
        
        base = 1
        digit = 0
        for i in reversed(range(len(new_s))):
            if new_s[i].isdigit():
                digit += int(new_s[i]) * base
                base *= 10
            else:
                break
        digit = digit * -1 if sign == "-" else digit
        if digit > MAX_INT: return MAX_INT
        if digit < MIN_INT: return MIN_INT
        return digit
            
        
