import json
import requests
import sys 
import mailgunDatapoints as dp


# the api key, auth tuple, base url
MAILGUN_API_KEY = 'XXX'
AUTHORIZATION   = ("api", MAILGUN_API_KEY)
BASE_URL        = "https://api.mailgun.net/v3/"


# enum for requests
class RequestType:
    get    = "GET"
    post   = "POST"
    delete = "DELETE"


# check response and propagate exception if needed
def handle_request(url, request_type=RequestType.get, params=None, propagate=True):
    response, status_code = None, None
    try:
        if request_type == RequestType.get:
            response = requests.get(url=url, auth=AUTHORIZATION, params=params)
        elif request_type == RequestType.post:
            response = requests.post(url=url, auth=AUTHORIZATION, params=params)
        else: 
            response = requests.delete(url=url, auth=AUTHORIZATION, params=params)
        response.raise_for_status()
    except Exception as e:
        print(e)
        if response:
            status_code = response.status_code
        if propagate:
            raise e
    return (response, status_code)


# helper function to get mailgun domain
def acquire_domain():
    print("Beginning to determine domain...")
    url = BASE_URL + "domains"
    response = requests.get(url=url, auth=("api", MAILGUN_API_KEY))
    try:
        response.raise_for_status()
        data = json.loads(response.text)
        domain = data["items"][0]["name"]
        print("Parsed", domain)
        return domain
    except Exception as e:
        print(e)
        raise e

# constant for mailgun domain
DOMAIN = acquire_domain()
print("\n\n\n")


# get all the mailing lists the user belongs to
def access(identifier):
    url = BASE_URL + "lists/pages"
    response, _ = handle_request(url)
    all_mailing_lists = [i["address"] for i in json.loads(response.text)["items"]]
    mailing_lists_with_user = []
    # all_mailing_lists = all_mailing_lists[30:]
    for mailing_list in all_mailing_lists:
        url = BASE_URL + "lists/{}/members/{}".format(mailing_list, identifier)
        response, _ = handle_request(url, propagate=False)
        if "not found" in response.text: 
            print("Email {} not a member of {}".format(identifier, mailing_list))
        else: 
            mailing_lists_with_user.append(mailing_list)

    # unsure what difference between data/context should be in runIntegration.py
    # returning both for now, that way mailing lists can be passed to erasure
    return {"data" : mailing_lists_with_user, "context": mailing_lists_with_user} 


# create a mailing address with a given address
def create_mailing_list(address):
    url = BASE_URL + "lists"
    payload = { "address" : "{}@{}".format(address, DOMAIN) }
    response, _ = handle_request(url, request_type=RequestType.post, params=payload)
    print(response.text)


# remove the user from all mailing lists
def erasure(identifier, context):
    print("Preparing to remove {} from {} email lists".format(identifier, len(context)))
    deleted_count = 0
    for mailing_list in context:
        url = BASE_URL + "lists/{}/members/{}".format(mailing_list, identifier)
        response, _ = handle_request(url, request_type=RequestType.delete, propagate=False)
        if "member has been deleted" in response.text:
            deleted_count += 1
        else:
            print("Error deleting {} from {}... {}".format(identifier, mailing_list, response.text))
    print("Successfully deleted email {}/{} mailing lists...".format(deleted_count, len(context)))


# create mailing lists and seed users into them
def seed(identifier):
    raise NotImplementedError("Seed not implemented!")


# Modify this list to add the identifiers you want to use.
sample_identifiers_list = [
    'spongebob@transcend.io',
    'squidward@transcend.io',
    'patrick_star@transcend.io',
    'sandy_cheeks@transcend.io'
]

class ActionType:
    # Fetch data for a given identifier
    # from the remote system, e.g. Mailgun.
    Access = 'ACCESS'
    # Delete data for a given identifier
    # from the remote system.
    Erasure = 'ERASURE'
    # Seed data into the remote system
    # creatine a profile with the given identifier.
    Seed = 'SEED'

def verify_action_args(args):
    """
    Validate arguments.
    """
    valid_actions = [ActionType.Seed, ActionType.Erasure, ActionType.Access]
    if len(args) != 2:
        raise ValueError('This module accepts a single argument: python3 runIntegration.py <action>, where <action> can be one of: {}'.format(", ".join(valid_actions)))
    action = args[1]
    if action not in valid_actions:
        raise ValueError("Action argument must be one of {}".format(", ".join(valid_actions)))
    return action


def run_integration(identifier, action_type):
    """
    Run the ACCESS and/or ERASURE flows for the given identifier.
    """
    print('Running access...\n')
    access_result = dp.access(identifier)
    data = access_result['data']
    print('Data retrieved for ' + identifier + ':')
    print(json.dumps(data, indent=2))

    if action_type == ActionType.Access:
        return

    context = access_result['context']
    print('Context for the erasure: ', context)
    print('\nRunning erasure...')
    dp.erasure(identifier, context)
    print('All done!')


def main():
    action = verify_action_args(sys.argv)
    data = sample_identifiers_list

    # Run the functions for all the identifiers we want to test
    for identifier in data:
        if action == ActionType.Seed:
            dp.seed(identifier)
        elif action == ActionType.Access or action == ActionType.Erasure:
            run_integration(identifier, action)
    return

if __name__ == "__main__":
    main()
