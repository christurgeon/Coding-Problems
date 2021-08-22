class Solution:
    
    # def majorityElement(self, nums: List[int]) -> int:
    #     nums.sort()
    #     return nums[len(nums) // 2]
    
    # def majorityElement(self, nums: List[int]) -> int:
    #     d = dict()        
    #     for n in nums:
    #         d[n] = d.get(n, 0) + 1
    #         if d[n] > len(nums)//2:
    #             return n
            
    def majorityElement(self, nums: List[int]) -> int:
        candidate, count = nums[0], 0
        for num in nums:
            if num == candidate:
                count += 1
            elif count == 0:
                candidate, count = num, 1
            else:
                count -= 1
        return candidate
