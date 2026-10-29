"""Pure independent-gate status predicate; never launches RT."""
def require_case_success(case,summary):
    if summary['counts']!=dict(Passed=1,Failed=0,Unknown=0):raise RuntimeError('Required independent CSI state not Passed: '+str(summary['counts']))
    if case.get('path_gate')=='Required_Passed' and summary['path_counts']!=dict(Passed=1,Partial=0,Failed=0,Unknown=0):
        raise RuntimeError('Required complete-path layer not Passed; stop before later GPU stages: '+str(summary['path_counts']))
