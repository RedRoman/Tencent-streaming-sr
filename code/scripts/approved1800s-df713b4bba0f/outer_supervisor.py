import json,os,resource,signal,subprocess,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
cfg=json.loads('{"project": "/home/ma-user/modelarts/user-job-dir/sol_h3", "ledger": "/home/ma-user/modelarts/user-job-dir/sol_h3/results/approved1800s_execution_df713b4bba0f.wall.json", "log": "/home/ma-user/modelarts/user-job-dir/sol_h3/results/approved1800s_execution_df713b4bba0f.outer.log", "argv": ["/home/ma-user/modelarts/user-job-dir/sol_h3/.venv/bin/python", "-B", "/home/ma-user/modelarts/user-job-dir/sol_h3/scripts/approved1800s-df713b4bba0f/probe_primitive_30min_30778.py", "--project-root", "/home/ma-user/modelarts/user-job-dir/sol_h3", "--runtime-root", "/home/ma-user/modelarts/user-job-dir/sol_h3/software/cann-8.5.2-910b-project", "--opp-root", "/home/ma-user/modelarts/user-job-dir/sol_h3/software/cann-8.5.2-910b-project/opp", "--api-root", "/home/ma-user/modelarts/user-job-dir/sol_h3/software/cann-8.5.2-910b-project", "--environment-helper", "/home/ma-user/modelarts/user-job-dir/sol_h3/scripts/probe_repaired_ops_isolated_30778.py", "--cpu-preflight", "/home/ma-user/modelarts/user-job-dir/sol_h3/results/approved1800s_cpu_gate_df713b4bba0f.json", "--authorization", "/home/ma-user/modelarts/user-job-dir/sol_h3/results/approved1800s_authorization_df713b4bba0f.json", "--output", "/home/ma-user/modelarts/user-job-dir/sol_h3/results/approved1800s_execution_df713b4bba0f.json", "--execute", "--authorization-token", "Sentinel_6ed0232ab588819191b34034cd6ad902"]}')
ledger_path=Path(cfg['ledger']);log_path=Path(cfg['log'])
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
ledger=json.loads(ledger_path.read_text())
def save():
    temporary=ledger_path.with_name(ledger_path.name+'.tmp')
    temporary.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(temporary,ledger_path)
started=time.monotonic();utc=datetime.now(timezone.utc)
ledger.update(phase='starting_global_guard',started_utc=utc.isoformat(),
              latest_allowed_finish_utc=(utc+timedelta(seconds=1800)).isoformat(),
              supervisor_pid=os.getpid(),outer_TERM_seconds=1798,outer_KILL_after_seconds=2)
save();process=None
try:
    with log_path.open('x') as stream:
        process=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=2s','1798s',*cfg['argv']],
                                 cwd=cfg['project'],stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        ledger.update(phase='running',outer_guard_pid=process.pid,outer_guard_pgid=process.pid);save()
        try:status=process.wait(timeout=1801)
        except subprocess.TimeoutExpired:
            ledger['outer_guard_wait_timeout']=True
            os.killpg(process.pid,signal.SIGKILL)
            status=process.wait(timeout=2)
    ledger.update(phase='complete',outer_guard_exit=status)
except BaseException as error:
    ledger.update(phase='supervisor_error',error_type=type(error).__name__,error=str(error))
    if process is not None and process.poll() is None:
        try:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
        except BaseException as cleanup_error:ledger['cleanup_error']=str(cleanup_error)
finally:
    ledger.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                  complete_guard_wall_seconds=time.monotonic()-started)
    ledger['within_1800s_wall']=ledger['complete_guard_wall_seconds']<=1800
    save()
