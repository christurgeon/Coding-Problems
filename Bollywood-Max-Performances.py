from collections import OrderedDict

def maxPerformances(arrivals, durations):
    idx = list(range(len(arrivals)))
    finish_times = [arrivals[i] + durations[i] for i in range(len(durations))]
    idx.sort(key = lambda x: finish_times[x])
    s = set()
    previous_finish = 0
    for i in idx:
        if arrivals[i] >= previous_finish:
            s.add(i)
            previous_finish = finish_times[i]
    return len(s)
