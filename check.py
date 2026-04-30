import sys, json
data = json.load(sys.stdin)
print('keys:', list(data.keys()))
print('total_rps:', data.get('total_rps'))
agg = (data.get('stats') or [{}])[-1]
print('agg name:', agg.get('name'))
print('agg keys:', list(agg.keys()))
print('p99:', agg.get('response_time_percentile_0.99'))