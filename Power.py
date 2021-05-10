class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        result = self.helper(x, x, n)
        if n != abs(n):
            return 1.0 / result
        else:
            return result

    def helper(self, orig, x, n):
        an = abs(n)
        if an == 0:
            return 1.0
        if an == 1:
            return x
        new_n = n-1 if n > 1 else n+1
        return self.helper(orig, orig*x, new_n)


    def myPowV2(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if x == 0:
            return 1.0
        result = 1.0
        for i in range(0, abs(n)):
            result *= x
        if n != abs(n):
            return 1.0 / result
        else:
            return result