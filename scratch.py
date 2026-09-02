import urllib.request
import json
try:
    req = urllib.request.Request('http://127.0.0.1:5001/api/samples')
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())
        for sample in data['samples'][:5]:
            print(sample['title'])
except Exception as e:
    print(e)
