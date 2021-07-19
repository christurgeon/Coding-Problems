class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals
        
        intervals = sorted(intervals, key=lambda i: i[0])
        overlapped = [intervals[0]]
        for i in range(1, len(intervals)):
            l, r = intervals[i][0], intervals[i][1]
            if overlapped[-1][1] >= l:
                overlapped[-1][1] = max(overlapped[-1][1], r)
            else:
                overlapped.append([l, r])
            
        return overlapped
            
            
        