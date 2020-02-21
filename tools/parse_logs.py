import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd


def parse_log(log_path: Path):
    print('Parsing log: {}'.format(log_path))

    log_id = '{}_{}'.format(
        log_path.name.replace('.log', ''),
        datetime.fromtimestamp(log_path.lstat().st_mtime).strftime('%Y-%m-%d-%H-%M-%S-%f')
    )
    train_data = []
    meta_data = {
        'log_id': log_id
    }
    with open(str(log_path), 'r') as f:
        line = f.readline()
        while line:
            # print(' - read line: {}'.format(line))
            matches = re.match(r'([a-z_0-9]+)=(.*)', line)
            if matches:
                # Meta data line
                name, value = matches.groups()
                meta_data[name] = value
            else:
                matches = re.match(r'([^,]+,\d+):Epoch \[(\d+)\]:\s([^=]+)=(.+)', line)
                if matches:
                    timestamp, epoch, metric_key, metric_val = matches.groups()
                    if metric_key != 'Best-Acc':
                        # print('  Extracted: {} {} {} {}'.format(timestamp, epoch, metric_key, metric_val))
                        train_data.append({
                            'log_id': log_id,
                            'timestamp': timestamp,
                            'epoch': epoch,
                            'metric_key': metric_key,
                            'metric_val': metric_val
                        })
            line = f.readline()
    return meta_data, train_data


def main():
    log_dir = Path('./log')

    if not log_dir.exists():
        print('Log directory does not exist.')
        return
    if not log_dir.is_dir():
        print('Log directory is not a directory.')
        return

    meta_list = []
    train_list = []
    for log_path in log_dir.glob('*/**/*.log'):
        meta_data, train_data = parse_log(log_path)
        meta_list.append(meta_data)
        train_list.extend(train_data)

    pd.DataFrame(meta_list).to_csv(str(log_dir / 'meta_data.csv'))
    pd.DataFrame(train_list).to_csv(str(log_dir / 'train_data.csv'))


if __name__ == '__main__':
    main()
