def hackerCards(collection, d):
    result = []
    collection = sorted(collection)
    previous = 0
    for i in range(len(collection)):
        diff = collection[i] - previous
        if diff > 1:
            for j in range(1, diff):
                new_card = previous + j
                result.append(new_card)
                d -= new_card
                if d <= new_card:
                    return result    
        previous = collection[i] 
    next_card = max(collection[-1], result[-1] if len(result) > 0 else -1) + 1
    while d >= next_card:
        result.append(next_card)
        next_card, d = next_card + 1, d - next_card
    return result