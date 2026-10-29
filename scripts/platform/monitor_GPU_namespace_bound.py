#!/usr/bin/env python3
"""CPU-only resource guard without guessing a host PID from Linux Ngid.

NVML can report host PIDs outside our procfs namespace. The maximum of ALL
reported compute processes on the approved UUID bounds any one reported
process, without attributing its allocation to our worker. A conservative
bound trip stops only the command/token/start-time-verified owned worker.
Unreported processes and gaps remain Unknown; no CUDA context is created.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def bound(rows, gpu_uuid, cap):
    selected = [r for r in rows if r['gpu_uuid'] == gpu_uuid]
    if not selected or any(r['used_bytes'] is None for r in selected):
        return dict(status='Unknown_missing_NVML_process_memory', upper_bound_bytes=None,
                    stop_owned_worker=False)
    maximum = max(r['used_bytes'] for r in selected)
    return dict(status='Observed_reported_process_set_upper_bound',
                upper_bound_bytes=maximum, stop_owned_worker=maximum > cap)


def parse_rows(output):
    result = []
    for line in output.splitlines():
        columns = [c.strip() for c in line.split(',')]
        if len(columns) != 3 or not columns[0].isdigit():
            continue
        pid, memory, uuid = columns
        result.append(dict(host_NVML_pid=int(pid), gpu_uuid=uuid,
                           used_bytes=int(memory)*1024**2 if memory.isdigit() else None))
    return result


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True)+'\n')
    temporary.replace(path)


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def verified_worker(job, pid):
    proc = Path('/proc', str(pid))
    command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    environment = (proc/'environ').read_bytes().split(b'\0')
    digest = hashlib.sha256((job/'config.json').read_bytes()+
                            (job/'effective_limits.json').read_bytes()).hexdigest()
    fields = (proc/'stat').read_text().split(') ', 1)[1].split()
    valid = ('generate --job '+str(job)+' ') in command and \
            ('WIWO_PLATFORM_WORKER_TOKEN='+digest).encode() in environment
    return valid, fields[19]  # Linux stat field22: start time, prevents PID reuse.


def main():
    p = argparse.ArgumentParser()
    for name in ('job', 'grant', 'orchestrator', 'output'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    job = Path(args.job).resolve()
    grant = json.loads(Path(args.grant).read_text())
    owner = json.loads(Path(args.orchestrator).read_text())['pid']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    samples = 0
    peak_bound = 0
    allocator_peak = 0
    unknown = 0
    started = time.time()
    with (out/'samples.jsonl').open('x') as log:
        while alive(owner):
            row = dict(unix_time=time.time(), owned_NVML_process_identity='Unknown',
                       Ngid_is_not_a_process_namespace_mapping=True)
            try:
                pid = json.loads((job/'worker_pid.json').read_text())['pid']
                valid, start = verified_worker(job, pid)
                row.update(local_worker_pid=pid, worker_start_ticks=start,
                           command_token_verified=valid)
                fds = []
                for fd in Path('/proc', str(pid), 'fd').iterdir():
                    try:
                        fds.append(os.readlink(fd))
                    except FileNotFoundError:
                        pass
                device = '/dev/nvidia'+str(grant['gpu'])
                row['approved_GPU_fd_observed'] = device in fds
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    timeout=4, check=True)
                rows = parse_rows(result.stdout)
                row['approved_UUID_processes'] = [r for r in rows if r['gpu_uuid'] == grant['gpu_uuid']]
                row.update(bound(rows, grant['gpu_uuid'], grant['max_allocator_bytes']))
                if not valid or device not in fds:
                    row.update(status='Unknown_owned_worker_or_device_not_confirmed',
                               stop_owned_worker=False)
                if row['status'] == 'Observed_reported_process_set_upper_bound':
                    samples += 1
                    peak_bound = max(peak_bound, row['upper_bound_bytes'])
                else:
                    unknown += 1
                if (job/'progress.json').exists():
                    progress = json.loads((job/'progress.json').read_text())
                    allocator_peak = max(allocator_peak, progress.get('allocator_peak_bytes', 0))
                    row['reported_worker_allocator_peak_bytes'] = allocator_peak
                    row['committed_frames'] = progress.get('committed_frames')
                if row['stop_owned_worker']:
                    # No host PID is signalled. Revalidate start time immediately.
                    still_valid, current_start = verified_worker(job, pid)
                    if still_valid and start == current_start:
                        row['action'] = 'SIGTERM_owned_worker_conservative_NVML_bound_trip'
                        os.kill(pid, signal.SIGTERM)
                        atomic(job/'control.json', dict(kind='stop', reason=row['action']))
                        atomic(out/'conservative_bound_trip.json', row)
            except (FileNotFoundError, ProcessLookupError, subprocess.SubprocessError) as exc:
                row.update(status='Unknown_observation_gap', error=repr(exc))
                unknown += 1
            log.write(json.dumps(row)+'\n')
            log.flush()
            atomic(out/'latest.json', row)
            time.sleep(2)
    atomic(out/'report.json', dict(
        status='Observed_reported_set_bound_only' if samples else 'Unknown',
        started_unix=started, ended_unix=time.time(), bound_samples=samples,
        unknown_samples=unknown, reported_process_set_peak_upper_bound_bytes=peak_bound or None,
        worker_allocator_reported_peak_bytes=allocator_peak or None,
        approved_cap_bytes=grant['max_allocator_bytes'], owned_NVML_process_identity='Unknown',
        limitation='Set bound is not owned-process measurement; unreported contexts, between-sample peaks and observation gaps remain Unknown. Ngid never used as PID evidence.'))


if __name__ == '__main__':
    main()
