class Solution:
    
    """
    each number appears twice except for 1 which appears once, find it
    """
    
    def singleNumber1(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num
        return result

    def singleNumber2(self, nums: List[int]) -> int:
        d = dict()
        for n in nums:
            d[n] = d.get(n, 0) + 1
            if d[n] == 2:
                del d[n]
        return list(d.keys())[0]
        
