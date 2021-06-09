import os
import numpy as np

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    logs = (i for i in os.listdir(dpath) if i.startswith('events.out.tfevents'))
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in logs
    ]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[EA.Scalars(tag) for EA in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath, outname):
    # logs = os.listdir(dpath)    # log filenames

    d, _ = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = [np.array(i).flatten().tolist() for i in values]
    max_step = len(max(np_values, key=len))

    for i, l in enumerate(np_values):
        np_values[i] = [str(j) for j in l] + [''] * (max_step - len(l))

    with open(outname, 'w') as f:
        writeline = lambda it: f.write(','.join(it) + '\n')
        writeline(('step', *tags))
        for i, it in enumerate(zip(*np_values)):
            writeline((str(i), *it))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path = "E:/Desktop/log/0601/ynet/detach"
    to_csv(path, 'E:/Desktop/0601-ynet-detach.csv')
