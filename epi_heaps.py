import heapq
import itertools


def topKLargest(k: int, stream: Iterator[str]):
    """
    Given a stream of elements, return the top k longest
    """
    min_heap = [(len(s), s) for s in itertools.islice(stream, k)] # store size and string
    heapq.heapify(min_heap)
    for next_string in stream:
        # push the next one and pop the smallest one
        heapq.heappushpop(min_heap, (len(next_string), next_string))
    return [s[1] for s in heapq.nsmallest(k, min_heap)]


def mergeSortedArrays(sorted_arrays: List[List[int]]):
    """
    Given a list of sorted arrays, merge them into a single list
    """
    min_heap = []
    sorted_arrays_iters = [iter(x) for x in sorted_arrays] 

    # put first element and index in the min_heap from each array
    for i, it in enumerate(sorted_arrays_iters):
        first = next(it, None) 
        if first is not None:
            heapq.heappush(min_heap, (first, i))

    result = []
    while min_heap:
        smallest, smallest_array_idx = heapq.heappop(min_heap)
        smallest_array_iter = sorted_arrays_iters[smallest_array_idx]
        result.append(smallest)
        next_elem = next(smallest_array_iter, None)
        if next_elem is not None:
            heapq.heappush(min_heap, (next_elem, smallest_array_idx))
    return result