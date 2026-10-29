"""Canonical absolute rigid-mesh arithmetic, independent of query history.

SDK SceneObject position/orientation setters transform current Float32 vertices.
For long trajectories or restored branches, callers may instead bind immutable
local vertices and form each world mesh once, casting only the final result.
The opt-in adapter updates the actual SDK scene/BVH. Mesh-history and restore
contracts are tested separately from CSI on this new geometry definition.
"""
import numpy as np


def absolute_vertices(local_vertices, position, rotation):
    local=np.asarray(local_vertices,float);p=np.asarray(position,float);r=np.asarray(rotation,float)
    if local.ndim!=2 or local.shape[1]!=3 or p.shape!=(3,) or r.shape!=(3,3):
        raise ValueError('Canonical local Nx3 vertices and rigid pose required')
    if not np.isfinite(local).all() or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Finite canonical geometry and pose required')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(r),1.,atol=1e-10,rtol=0):
        raise ValueError('Proper orthonormal rotation required')
    return (local@r.T+p).astype(np.float32)


def apply_absolute_pose(engine, name, local_vertices, position, rotation):
    """Pinned SDK 2.2.0 application; installed SDK and V4 remain untouched.

    Explicit canonical local vertices are required. Never infer an undeformed
    rest mesh from a current, already accumulated Float32 mesh.
    """
    import drjit as dr
    import mitsuba as mi
    from scipy.spatial.transform import Rotation
    entry=engine.dynamic_actors[name];obj=entry['object']
    world=absolute_vertices(local_vertices,position,rotation)
    old=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
    if old.shape!=world.shape:raise ValueError('Canonical vertex topology differs')
    # The SDK helper updates scene parameters, BVH and scene_geometry_updated().
    # Its closure intentionally ignores accumulated current vertices.
    obj._transform_vertices(lambda _:mi.Point3f(world.T))
    obj._orientation=mi.Point3f(Rotation.from_matrix(rotation).as_euler('ZYX'))
    entry['position']=np.asarray(position,float).copy();entry['rotation']=np.asarray(rotation,float).copy()
    dr.eval();dr.sync_thread();engine.mark_environment_changed()
    return world


def install_absolute_actor_updater(engine, rest_meshes):
    """Explicitly opt an owner into history-independent actual actor geometry.

    Every actor needs supplied immutable local rest vertices. Topology/object
    replacement is rejected; all requested poses are validated before mutation.
    Existing V4 setters and past datasets retain their original definitions.
    Owner snapshots use this updater automatically when restoring actor poses.
    """
    from types import MethodType
    from .native_path_geometry import array_hash
    if hasattr(engine,'_v5_absolute_actor_rest'):
        raise ValueError('Absolute actor updater already installed')
    if set(rest_meshes)!=set(engine.dynamic_actors):
        raise ValueError('Explicit rest mesh required for every registered actor')
    bound={}
    for name,vertices in rest_meshes.items():
        entry=engine.dynamic_actors[name];obj=entry['object']
        local=np.asarray(vertices,float).copy();local.setflags(write=False)
        current=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
        if not np.array_equal(current,absolute_vertices(local,entry['position'],entry['rotation'])):
            raise ValueError('Bind before motion: current mesh differs from supplied canonical rest pose')
        bound[name]={'local':local,'object':obj,'topology':array_hash(np.asarray(obj.mi_mesh.faces_buffer()))}
    engine._v5_absolute_actor_rest=bound
    def update(owner,updates):
        import drjit as dr
        import mitsuba as mi
        from scipy.spatial.transform import Rotation
        from .native_path_geometry import array_hash
        if set(owner.dynamic_actors)!=set(bound):raise ValueError('Registered actor set changed; explicit rebind required')
        prepared=[]
        for name,pose in updates.items():
            if name not in bound:raise KeyError('Unknown canonical actor '+name)
            rest=bound[name];entry=owner.dynamic_actors[name];obj=entry['object']
            if obj is not rest['object'] or array_hash(np.asarray(obj.mi_mesh.faces_buffer()))!=rest['topology']:
                raise ValueError('Bound actual actor topology/object changed')
            p=np.asarray(pose.get('position',entry['position']),float)
            r=np.asarray(pose.get('rotation',pose.get('rotation_matrix',entry['rotation'])),float)
            world=absolute_vertices(rest['local'],p,r)
            current=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
            # Small requested motion remains causal even if its final F32 mesh
            # is unchanged; never introduce a tolerance-based motion deadband.
            if not np.array_equal(p,entry['position']) or not np.array_equal(r,entry['rotation']) or not np.array_equal(world,current):
                prepared.append((name,obj,p.copy(),r.copy(),world))
        changed=[]
        try:
            for name,obj,p,r,world in prepared:
                obj._transform_vertices(lambda _,vertices=world:mi.Point3f(vertices.T))
                obj._orientation=mi.Point3f(Rotation.from_matrix(r).as_euler('ZYX'))
                owner.dynamic_actors[name]['position']=p;owner.dynamic_actors[name]['rotation']=r;changed.append(name)
        finally:
            if prepared:
                # Any GPU-side failure also invalidates propagation banks;
                # never expose a stale epoch over partially changed geometry.
                dr.eval();dr.sync_thread();owner.mark_environment_changed()
        if changed:owner.stats['actual_dynamic_geometry_updates']+=1
        return {'changed':changed,'scene_version':owner.environment_version,'actual_bvh_updated':bool(changed),
            'pose_model':'absolute_immutable_rest_mesh_Float64_then_Float32'}
    engine.update_dynamic_actors=MethodType(update,engine)
    return {'status':'Implemented_opt_in_absolute_actor_geometry','actors':sorted(bound),
        'rest_mesh_sha256':{name:array_hash(x['local']) for name,x in bound.items()},
        'CSI_validation':'Unknown_until_independent_original_Full_on_matching_actual_geometry'}
