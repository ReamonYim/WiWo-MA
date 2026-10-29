"""One explicit operator lease; observers cannot mutate shared form/state.

Operational serialization for a trusted localhost/SSH service, not tenant
authentication. The pinned Viser ingress guard runs before shared GUI mutation.
"""
import json,secrets,threading,time
from pathlib import Path


class ControlLease:
    def __init__(self,log=None):
        self.lock=threading.RLock();self.owner=None;self.clients=set();self.revision=0;self.events=[];self.ingress_rejections=0;self.log=None if log is None else Path(log)
    def record(self,kind,client,**extra):
        row=dict(time_ns=time.time_ns(),kind=kind,client='api' if isinstance(client,str) and client.startswith('api:') else client,revision=self.revision,**extra)
        self.events.append(row);self.events=self.events[-512:]
        if kind=='ingress_rejected':self.ingress_rejections+=1
        if self.log:
            self.log.parent.mkdir(parents=True,exist_ok=True)
            with self.log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
    def connect(self,client):
        with self.lock:
            self.clients.add(client)
            if self.owner is None:self.owner=client
            self.record('connect',client)
    def disconnect(self,client):
        with self.lock:
            self.clients.discard(client)
            if self.owner==client:self.owner=None;self.revision+=1
            self.record('disconnect_no_worker_interruption',client)
    def claim(self,client):
        with self.lock:
            if self.owner not in (None,client):
                self.record('claim_rejected',client);raise PermissionError('Another operator owns the shared platform; current operator must release control')
            self.owner=client;self.revision+=1;self.record('claim',client);return self.view()
    def release(self,client):
        with self.lock:
            self.require(client);self.owner=None;self.revision+=1;self.record('release_no_worker_interruption',client)
    def require(self,client,revision=None):
        with self.lock:
            if client is None or client!=self.owner:
                self.record('observer_mutation_rejected',client);raise PermissionError('Observer is read-only; explicitly acquire control first')
            if revision is not None and revision!=self.revision:
                self.record('stale_revision_rejected',client,expected=revision);raise ValueError('Stale platform revision; reload current state before retrying')
    def mutation(self,client,kind):
        with self.lock:self.require(client);self.revision+=1;self.record(kind,client)
    def api_claim(self):
        session='api:'+secrets.token_hex(24);self.claim(session);return session
    def view(self):
        with self.lock:return dict(owner_client_id=self.owner if isinstance(self.owner,int) else None,owner_kind='API_operator' if isinstance(self.owner,str) else 'browser_operator' if self.owner is not None else 'unclaimed',ingress_rejections=self.ingress_rejections,
            revision=self.revision,connected_clients=sorted(self.clients),observer_policy='Read-only shared scene; no input, Action, task or replay mutation without control lease')


class ViserLeaseBridge:
    """Pinned Viser1.1.1 protocol adapter, guarded by real browser tests.

    Reject messages BEFORE GuiApi mutates/broadcasts a shared input value;
    a button-only permission check cannot prevent observer form poisoning.
    Camera/navigation stays local. No SDK/GPU functionality is used here.
    """
    def __init__(self,ui):
        import asyncio
        import importlib.metadata
        if importlib.metadata.version('viser')!='1.1.1':raise RuntimeError('Control lease ingress requires validated Viser1.1.1')
        from viser import _messages as m
        self.ui=ui;self.lease=ui.sessions;self.labels={};self.m=m
        self._input_order=asyncio.Lock()
        interface=ui.server._websock_server
        classes=[m.GuiUpdateMessage,m.TransformControlsUpdateMessage,m.TransformControlsDragStartMessage,m.TransformControlsDragEndMessage,m.SceneNodeClickMessage]
        for cls in classes:
            for original in list(interface._incoming_handlers.get(cls,[])):
                interface.unregister_handler(cls,original)
                async def guarded(client,message,original=original,cls=cls):
                    # Client-local lease controls are handled by client.gui.
                    handle=ui.server.gui._gui_input_handle_from_uuid.get(getattr(message,'uuid',None))
                    if cls is m.GuiUpdateMessage and handle is None:return await original(client,message)
                    if cls is not m.GuiUpdateMessage and getattr(message,'owner','')!='':return await original(client,message)
                    # Serialize the input write itself with scene/config form
                    # reconstruction, not only the later button callback.
                    # Otherwise new target values can be overwritten midway
                    # through a configuration load and accepted as a no-op.
                    # Critical platform callbacks are awaited, rather than
                    # detached into Viser's pool. Tasks for successive wire
                    # messages also take this FIFO async lock.
                    async with self._input_order:
                        with ui.render_lock:
                            try:self.lease.require(client)
                            except PermissionError:
                                self.lease.record('ingress_rejected',client,message_type=cls.__name__,target=getattr(message,'uuid',getattr(message,'name',None)))
                                if handle is not None:
                                    c=ui.server.get_clients().get(client)
                                    if c:c._websock_connection.queue_message(m.GuiUpdateMessage(handle._impl.uuid,dict(value=handle.value,disabled=True)))
                                return
                            result=await original(client,message)
                            with self.lease.lock:
                                if self.lease.owner==client:self.lease.mutation(client,'accepted_'+cls.__name__)
                                else:self.lease.record('accepted_input_before_lease_change',client)
                            return result
                interface.register_handler(cls,guarded)
    def refresh(self):
        for client_id,client in self.ui.server.get_clients().items():
            owner=self.lease.owner==client_id
            if client_id in self.labels:self.labels[client_id].content=f"本浏览器：**{'操作员' if owner else '观察者（只读）'}**；版本{self.lease.revision}。断线不停止后台任务。"
            for uuid,handle in self.ui.server.gui._gui_input_handle_from_uuid.items():
                client._websock_connection.queue_message(self.m.GuiUpdateMessage(uuid,dict(disabled=bool(getattr(handle,'disabled',False)) if owner else True)))
    def connect(self,client):
        self.lease.connect(client.client_id)
        with client.gui.add_folder('会话权限（每个浏览器独立）',expand_by_default=True,order=-100):
            self.labels[client.client_id]=client.gui.add_markdown('正在核验会话…')
            claim=client.gui.add_button('申请平台操作权');release=client.gui.add_button('释放平台操作权')
        def action(fn):
            def callback(event):
                try:
                    with self.ui.render_lock:fn(client.client_id);self.refresh()
                except (PermissionError,ValueError) as exc:self.labels[client.client_id].content='**操作被拒绝：** '+str(exc)
            return callback
        claim.on_click(action(self.lease.claim));release.on_click(action(self.lease.release));self.refresh()
    def disconnect(self,client):
        with self.ui.render_lock:
            self.lease.disconnect(client.client_id);self.labels.pop(client.client_id,None);self.refresh()
