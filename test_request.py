import urllib.request
import json

url = 'http://127.0.0.1:5000/api/returnresult'
req = urllib.request.Request(url, data=b'{}', headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print('STATUS', resp.status)
        print(resp.read().decode())
except Exception as e:
    # If the server returned an HTTP error (e.g. 500), urllib raises HTTPError
    # which has a `.read()` method for the response body — print that to see
    # any JSON error details returned by the Flask route.
    try:
        body = e.read().decode()
        print('HTTP ERROR BODY:', body)
    except Exception:
        print('ERROR', e)
