class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # 123 % 10 =
        new_digit = 0
        curr = abs(x)
        if x != 0:
            sign = int(curr / x)
        else:
            sign = 1
        while curr > 0:
            rem = curr % 10
            new_digit = 10*new_digit + rem
            curr = curr // 10
        return sign*new_digit
        
            