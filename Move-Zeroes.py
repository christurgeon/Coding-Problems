class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        chunk_size = i = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                chunk_size += 1
            elif chunk_size >= 1:
                nums[i], nums[i-chunk_size] = nums[i-chunk_size], nums[i]
