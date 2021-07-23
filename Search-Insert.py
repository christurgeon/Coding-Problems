class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, h = 0, len(nums) - 1
        if target < nums[0]:
            return 0
        if target > nums[-1]:
            return len(nums)
        while l <= h:
            mid = (l + h) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target and nums[mid+1] > target:
                return mid+1
            if nums[mid] > target:
                h = mid - 1
            else:
                l = mid + 1
