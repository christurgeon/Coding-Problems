class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        max_so_far = float('-inf')
        tracker = dict()

        for idx, num in enumerate(nums):
            if num in tracker:
                count, start, end = tracker[num]
                tracker[num] = (count + 1, start, idx + 1)
                current_max = count + 1
            else:
                tracker[num] = (1, idx, idx + 1) 
                current_max = 1
            max_so_far = max(max_so_far, current_max)
        
        start, end = 0, len(nums)
        for k, v in tracker.items():
            count, element_start, element_end = v
            if count == max_so_far:
                diff = element_end - element_start
                if diff < end - start:
                    start, end = element_start, element_end
            
        return end - start
            
                
