class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = dict()
        for i, num in enumerate(nums):
            if num in d:
                return [d[num], i]
            else:
                diff = target - num
                d[diff] = i
        