class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True

        max_can_reach = nums[0]
        current_index = 0
        while max_can_reach < len(nums) - 1 and current_index != max_can_reach:

            max_range = max_can_reach
            for i in range(current_index, max_range + 1):
                max_can_reach = max(max_can_reach, i + nums[i])

            current_index = max_range

        return True if max_can_reach >= len(nums) - 1 else False
