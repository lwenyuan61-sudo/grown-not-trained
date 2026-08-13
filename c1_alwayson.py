'''C1 rig under the final always-on config: single food action, body vs sham.'''
import json
from neo_core import P
from value_ladder1 import run_e1, KBODY

if __name__ == "__main__":
    P['alwaysOn'] = 1; P['datPump'] = 1
    for seed in (101, 103, 107, 109, 113, 127, 131, 137, 139, 149):
        body = run_e1(seed, KBODY); sham = run_e1(seed, 0.0)
        print(json.dumps(dict(seed=seed, body=body['final'], sham=sham['final'],
                              PASS=bool(body['final'] >= 0.8 and sham['final'] <= 0.35))),
              flush=True)

