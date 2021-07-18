class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def search(path, used, result):
            if len(path) == len(nums):
                result.append(copy.deepcopy(path))
                return
            
            for idx, element in enumerate(nums):
                if used[idx]:
                    continue
                used[idx] = True
                path.append(element)
                search(path, used, result)
                
                # remove the newly added element and start another permutation
                path.pop()
                used[idx] = False
            
        result = []
        used = [False] * len(nums)
        search([], used, result)
        return result
        