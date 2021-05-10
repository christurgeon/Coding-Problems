class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if nums is None:
            return False
        length = len(nums)
        if length <= 2:
            return True
        changes = False
        for i in range(length-1):
            if nums[i] > nums[i+1]:
                if changes:
                    return False
                else:
                    changes = True
                if i > 0:
                    if nums[i-1] > nums[i+1]: 
                        nums[i+1] = nums[i]
        return True
            
            
    