class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        width = len(height) - 1
        result = 0
        for i in reversed(range(1, width + 1)):
            if height[left] >= height[right]:
                result = max(result, height[right] * i)
                right -= 1
            else:
                result = max(result, height[left] * i)
                left += 1
        return result
