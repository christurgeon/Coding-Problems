class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        current_max = float('-inf')
        subarray_max = nums[0]
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                subarray_max += nums[i]
            else:
                current_max = max(current_max, subarray_max)
                subarray_max = nums[i]
        return max(current_max, subarray_max)        
