
def popularPlace(data): # [(d, p, t), ...]
    counter = dict()
    for _, place, _ in data:
        if place in counter:
            counter[place] += 1
        else:
            counter[place] = 0
    
    place = None
    max_count = float('-inf')
    for place_id, count in counter:
        if count > max_count:
            place = place_id
            max_count = count

    return place


def distinctVisits(data):
    place_count = dict()
    device_at_place_tracker = dict()
    
    for device_id, place_id, timestamp in data:
        if place_id in device_at_place_tracker:
            
            devices = device_at_place_tracker[place_id]
            device_is_present = device_id in devices
            if device_is_present and timestamp - devices[device_id] > 10000:
                place_count[place_id] += 1
                
            elif device_is_present: 
                devices[device_id] = timestamp 
                
            else:
                device_at_place_tracker[place_id][device_id] = timestamp
                
        else:
            place_count[place_id] = 1
            device_at_place_tracker[place_id] = {device_id : timestamp}
             
                
            
