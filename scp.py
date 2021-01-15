import os
import re

paste = input('paste: ')
m = re.search(r'ssh -p (\d+) (\w+@\d+\.\d+\.\d+\.\d+)', paste)
port = m.group(1)
host = m.group(2)

os.system(
    'scp -r -P {port} ./src ./config .env {host}:/root/BUSIC'.format(port=port, host=host)
)
os.system(
    'scp -P {port} ./tmp/data/BIRADpt.zip {host}:/input'.format(port=port, host=host)
)