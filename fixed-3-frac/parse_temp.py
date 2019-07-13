import yaml
import sys

dir = sys.argv[1]

f = open(dir + '/heat_observe.yaml', 'r')
d = yaml.safe_load(f)
for i in d['data']:
  print(i['time'],i['temperature'][0])
