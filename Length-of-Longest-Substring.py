class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        cache = dict()
        size = 0
        for idx, i in enumerate(s):
            if i in cache:
                size = max(size, len(cache))
                starting = cache[i]
                cache = {k: v for k,v in cache.items() if v > starting}
            cache[i] = idx
                
        return max(size, len(cache))        