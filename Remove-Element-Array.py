class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = -1
        for j in range(0, len(nums)):
            if nums[j] != val:
                i += 1
                nums[i] = nums[j]
        return i + 1
        
