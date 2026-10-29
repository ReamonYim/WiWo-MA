"""Platform entrypoint. Planning/preview/verification do not initialize CUDA."""
import argparse,json,os
from pathlib import Path
from .config import ROOT,load,estimated_storage,fingerprint
from .storage import atomic_json


def main():
    p=argparse.ArgumentParser(prog='wiwo-ma-platform');s=p.add_subparsers(dest='command',required=True)
    s.add_parser('doctor')
    a=s.add_parser('plan');a.add_argument('--config',required=True);a.add_argument('--job');a.add_argument('--max-wall-s',type=float,default=300.);a.add_argument('--max-disk-gib',type=float,default=8.)
    a=s.add_parser('serve');a.add_argument('--config',required=True);a.add_argument('--port',type=int,default=8080);a.add_argument('--host',default='127.0.0.1');a.add_argument('--authorization')
    a=s.add_parser('replay');a.add_argument('--dataset',required=True);a.add_argument('--port',type=int,default=8080);a.add_argument('--host',default='127.0.0.1')
    a=s.add_parser('generate');a.add_argument('--job',required=True);a.add_argument('--resume',action='store_true');a.add_argument('--frame-limit',type=int);a.add_argument('--authorized-worker',action='store_true')
    a=s.add_parser('supervise');a.add_argument('--job',required=True);a.add_argument('--authorization',required=True);a.add_argument('--resume',action='store_true');a.add_argument('--frame-limit',type=int)
    a=s.add_parser('launch');a.add_argument('--job',required=True);a.add_argument('--authorization',required=True);a.add_argument('--resume',action='store_true');a.add_argument('--frame-limit',type=int)
    a=s.add_parser('verify');a.add_argument('--dataset',required=True)
    a=s.add_parser('prepare-worker');a.add_argument('--job',required=True)
    a=s.add_parser('audit-closed');a.add_argument('--job',required=True)
    a=s.add_parser('preflight');a.add_argument('--config',required=True);a.add_argument('--output',required=True)
    a=s.add_parser('examples');a.add_argument('--output',default='configs/platform')
    args=p.parse_args()
    if args.command=='doctor':
        from .environment import inspect_environment
        report=inspect_environment();print(json.dumps(report,indent=2))
        if report['status'].startswith('Failed'):raise SystemExit(2)
        return
    if args.command=='prepare-worker':
        from .workflow import prepare_worker
        prepare_worker(args.job);return
    if args.command=='audit-closed':
        from .workflow import audit_closed
        result=audit_closed(args.job);print(json.dumps(result,indent=2))
        if result['status']=='Failed':raise SystemExit(2)
        return
    if args.command=='preflight':
        from .scenario_checks import preflight
        result=preflight(load(args.config),args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('scene_resources','sampled_visibility','motion','violations')},indent=2))
        if result['status']!='Passed':raise SystemExit(2)
        return
    if args.command=='plan':
        c=load(args.config);report=dict(status='prepared_not_started',configuration_sha256=fingerprint(c),estimate=estimated_storage(c),limits=dict(max_wall_s=args.max_wall_s,max_disk_bytes=int(args.max_disk_gib*1024**3),max_allocator_bytes=12*1024**3),GPU_started=False)
        if args.job:
            from .jobs import prepare_job
            prepare_job(c,args.job,report['limits'])
        print(json.dumps(report,indent=2));return
    if args.command=='examples':
        from .examples import reflector_example,munich_example
        folder=(ROOT/args.output).resolve()
        if not folder.is_relative_to(ROOT):raise ValueError('Output outside project')
        folder.mkdir(exist_ok=True,parents=True)
        for name,c in [('reflector',reflector_example()),('munich64_k40_28ghz',munich_example()),('munich64_k35_24ghz',munich_example(35,2.4e9))]:
            path=folder/(name+'.json')
            if path.exists():raise FileExistsError(path)
            atomic_json(path,c)
        print(folder);return
    if args.command=='serve':
        from .web import serve
        serve(load(args.config),host=args.host,port=args.port,authorization=args.authorization);return
    if args.command=='replay':
        from .storage import CSIReader
        from .web import serve
        reader=CSIReader(args.dataset)
        if not reader.complete:raise ValueError('Replay command requires a closed complete dataset')
        serve(reader.config,host=args.host,port=args.port,replay=args.dataset);return
    if args.command=='generate':
        import hashlib
        job=(ROOT/args.job).resolve()
        expected=hashlib.sha256((job/'config.json').read_bytes()+(job/'effective_limits.json').read_bytes()).hexdigest() if (job/'effective_limits.json').exists() else None
        if not args.authorized_worker or expected is None or os.environ.get('WIWO_PLATFORM_WORKER_TOKEN')!=expected:raise ValueError('Use launch with an explicit independent native-platform authorization')
        from .jobs import generate
        generate(args.job,resume=args.resume,frame_limit=args.frame_limit);return
    if args.command in ('supervise','launch'):
        from .jobs import supervise,launch_job
        fn=supervise if args.command=='supervise' else launch_job
        try:result=fn(args.job,args.authorization,resume=args.resume,frame_limit=args.frame_limit)
        except BaseException as exc:
            if args.command=='supervise':atomic_json(Path(args.job)/'state.json',dict(status='failed_supervisor_preserved',error=repr(exc),no_automatic_retry=True))
            raise
        print(json.dumps(dict(result=result)))
        if args.command=='supervise' and result:raise SystemExit(result)
        return
    if args.command=='verify':
        from .storage import CSIReader
        reader=CSIReader(args.dataset)
        for i in range(reader.count):reader.frame(i,verify=True)
        print(json.dumps(dict(storage='Passed',frames=reader.count,complete=reader.complete,numerical_accuracy='Unknown_unless_independent_SDK_audit_present')))


if __name__=='__main__':main()
