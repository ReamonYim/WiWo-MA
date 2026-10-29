/* WiWo-MA numeric draft guard. Does not change or synthesize server values. */
(() => {
  const invalid = new Map();
  const numeric = el => el instanceof HTMLInputElement &&
    (el.type === 'number' || ['decimal', 'numeric'].includes(el.inputMode));
  const complete = raw => /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw.trim()) &&
    Number.isFinite(Number(raw));
  function show() {
    for (const el of invalid.keys()) if (!el.isConnected) invalid.delete(el);
    let banner = document.getElementById('wiwo-input-error');
    if (!invalid.size) { if (banner) banner.remove(); return; }
    if (!banner) {
      banner = document.createElement('div'); banner.id = 'wiwo-input-error';
      banner.setAttribute('role', 'alert'); banner.setAttribute('aria-live', 'assertive');
      Object.assign(banner.style, {position:'fixed',bottom:'12px',left:'12px',right:'12px',
        zIndex:'2147483647',background:'#7f1d1d',color:'white',padding:'12px',
        border:'2px solid #fca5a5',borderRadius:'6px',pointerEvents:'none'});
      document.body.appendChild(banner);
    }
    banner.textContent = '数字输入尚未完成或无效；未提交动作/配置。请重新填写完整有限数字（不能留空），再点击执行。';
  }
  document.addEventListener('input', event => {
    const el = event.target;
    if (!numeric(el)) return;
    // Capture the raw DOM text BEFORE React/Mantine normalizes it on blur.
    // A normalization is not a user's confirmation of an older server value.
    if (complete(el.value)) { invalid.delete(el); el.removeAttribute('aria-invalid'); }
    else { invalid.set(el, el.value); el.setAttribute('aria-invalid', 'true'); }
    show();
  }, true);
  const commits = /执行目标姿态|将所选目标加入联合动作|原子执行全部联合动作|执行有限时长速度命令|验证并应用|应用所选UE|应用actor|冻结当前|冻结配置|启动已准备任务|保存配置预设|下载当前完整实验配置/;
  function guard(event) {
    const button = event.target instanceof Element ? event.target.closest('button') : null;
    if (!button || !commits.test(button.textContent || '')) return;
    show();
    if (!invalid.size) return;
    event.preventDefault(); event.stopImmediatePropagation();
    invalid.keys().next().value.focus({preventScroll:true});
  }
  // pointerdown is before blur; click also covers keyboard/assistive activation.
  document.addEventListener('pointerdown', guard, true);
  document.addEventListener('click', guard, true);
})();
