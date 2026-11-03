# 本地 Town02 配置

本地默认示例使用由 `Sionna_Town02/Town2_xml/` 原始资源生成的
`assets/town02/town2_strict_itu.xml` 保守材质版本。
资源复制到 `assets/town02/`，原文件夹不作修改。导入器按资源包的 SHA-256 清单
核对文件，再检查复制结果。导入清单为 `assets/town02/import-manifest.json`。

资源包括全部 8,577 个 PLY、4 份 XML、53 份车辆轨迹 CSV 和原始材质管理脚本。
原三材质场景 `town2_3mat.xml` 引用其中 6,085 个 PLY：4,920 个绑定 `mat_brick`，551 个绑定
`mat_wood`，614 个绑定 `mat_floorboard`。XML 对应的 ITU 类型分别为
`brick`、`wood`、`concrete`。未应用全局材质覆盖或天气参数替换。
原三材质 XML 和配置继续保留。浏览器按无线材质分组着色；这些显示色不代表原始 CARLA/Unreal 贴图。

## 按原始标签匹配的材质版本

规则文件为 [`configs/materials/town02_strict_itu.json`](../configs/materials/town02_strict_itu.json)。
只按原始 `town2_xml.xml` 中的 **21 个明确材质 ID** 匹配，再通过完全相同的 PLY 文件名
追溯到三材质场景。没有使用颜色、PBR 数值、通配符或物体类别推断电磁材质。
例如 `Glass`、`Concrete`、`Wood` 的明确标签可以匹配；`Window` 不直接当玻璃，
铝不直接当通用金属，石头不当大理石，湿路面不当沥青，树叶和塑料不改成木材。
光学名称仍是资源作者的标签依据，不能当作实测物理成分。

408 个网格匹配到明确标签，其中 395 个绑定需要更正，13 个原绑定已与标签一致。
其余 **5,677 个网格的物理材质未核实**，保留旧绑定并单独标记，不能据此称整图材质都已确认。
输出仍是 6,085 个原网格，2,831,119 个顶点、3,268,710 个三角形，几何不变。

| Sionna ITU 类型 | 匹配明确标签的网格 | 最终绑定总数（含未核实旧绑定） | 2.8 GHz 相对介电常数 | 2.8 GHz 电导率 S/m |
| --- | ---: | ---: | ---: | ---: |
| `brick` | 41 | 4,601 | 3.91 | 0.02806223 |
| `concrete` | 187 | 753 | 5.24 | 0.10337344 |
| `wood` | 24 | 575 | 1.99 | 0.01416974 |
| `glass` | 91 | 91 | 6.31 | 0.01429638 |
| `metal` | 65 | 65 | 1.00 | 10,000,000 |

电磁参数来自本机安装的 **Sionna RT 2.2.0** `ITU_MATERIALS_PROPERTIES`，按 SDK 公式
`εr = a × f_GHz^b`、`σ = c × f_GHz^d` 计算。XML 仅指定 `itu-radio-material` 和内置类型，
没有手填介电常数、电导率、厚度、散射系数或天气修正值，保留 SDK 默认值。
修改载频时会检查所有使用类型的内置有效频段，超出范围拒绝加载。
这决定了材料模型，不代表已验证完整传播、透射或散射结果。

每个网格的来源、替换前后绑定和证据记录于
`assets/town02/town2_strict_itu.materials.json`；本次替换的逐 ID 便览表为
`artifacts/material-assignment/source-material-review.csv`，实际更改的 395 个网格列在
`artifacts/material-assignment/changed-meshes.csv`。
配置绑定 XML、规则、审计和 SDK 源码的 SHA-256；加载资源时还核对原始 8,635 个文件，
避免配置、材质或几何被无意改动后继续使用。Web 上会显示匹配数和未核实数。

```bash
.venv-full/bin/python -m wiwo_ma.platform.town02_materials check
# 仅在有意更新精确 ID 规则后重新生成独立 XML、审计和两个配置：
.venv-full/bin/python -m wiwo_ma.platform.town02_materials build
```

## 安装和启动

完整安装入口会准备独立的 `.venv-full`，安装项目运行依赖、测试工具与 Chromium，
导入 Town02，补齐原版文档图片，并执行依赖检查、CPU 测试及两个机械示例。
已有的 `.venv` 不会被覆盖。
安装需要可访问包下载源；未找到 Python 3.11 及以上时会尝试在项目内用 Conda 创建
Python 3.11 和 Git 环境。

```bash
cd WiWo-MA
scripts/platform/install
scripts/platform/town02 --port 8080
```

### 后台运行，以后直接打开网页

依赖和地图已安装好后，只需运行一次：

```bash
cd WiWo-MA
scripts/platform/town02-service restart
```

后台服务默认绑定 `0.0.0.0:8080`，API 使用 `8081`，加载保守材质版的 64TX / 40UE 配置。
通过服务器的 8080 端口网页入口访问；使用平台端口映射时，在平台中打开对应网页。
SSH 访问也可以转发 8080 和 8081。启动命令返回前会确认网页和 API 可用、地图已绘制，
并确认当前配置使用保守材质版。关闭网页或退出终端不会停止后台进程。

本地安装还在 `.vscode/settings.json` 配置了 VS Code Remote SSH 自动端口转发。
在远程窗口打开本项目时，8080 转发后尝试自动打开浏览器，8081 静默转发，重新连接时恢复端口。
若已有服务未被自动识别，在 VS Code 按 F1，运行 `Forward a Port`，输入 `8080`；
再为 `8081` 重复一次。在底部“端口 / Ports”面板找到 `WiWo-MA Town02`，点击“在浏览器中打开”。
本地通常访问 `http://localhost:8080`；若本地端口被占用，以该面板显示的地址为准。
通过这种方式访问时，需要保持 VS Code 的远程连接。

```bash
scripts/platform/town02-service status
scripts/platform/town02-service stop
```

日志和进程状态保存在 `.state/town02-service/`。管理脚本只停止自己记录并核实身份的
网页进程，不会结束其他程序或原生计算任务；端口被其他进程占用时会报告错误。
需要仅本机访问时，可用 `scripts/platform/town02-service restart --host 127.0.0.1`。

后台进程可以跨终端会话运行，**不能跨容器重启保留**。若希望容器启动后自动提供网页，
需在运行平台的容器启动设置中加入下面的前台命令，由平台维护其生命周期：

```bash
scripts/platform/town02 --host 0.0.0.0 --port 8080
```

已有兼容解释器时，可设置 `WIWO_PYTHON=/path/to/python3.11`。另一份资源包路径可用
`WIWO_TOWN02_SOURCE=/path/to/Sionna_Town02` 指定。Web 默认仅监听本机；SSH 需要转发
8080 和相邻的 8081 API 端口。

安装器检查 CPU 环境，不安装内核驱动，也不启动原生射线计算。
GPU 功能仍需要机器上可用的 NVIDIA 驱动和匹配的 OptiX，按
[GPU 工作流](gpu-jobs.md)执行实际求解。

Ubuntu 20.04 使用 `browser-focal` 额外依赖中的 Playwright 1.62.0；
其他系统使用 `browser` 中的 1.63.0。1.63 已取消 Ubuntu 20.04 支持。
浏览器工具仅用于服务器上的自动化截图和测试，平台网页可直接用客户端浏览器打开。
运行依赖及地图检查通过后即可启动预览，不必等待服务器端 Chromium 安装。

依赖已装好时，可分别运行下面的检查和浏览器安装，无须重装全部依赖：

```bash
scripts/platform/check
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple scripts/platform/install_browser
scripts/platform/town02 --port 8080
```

浏览器安装步骤还会实际启动 Chromium，检查软件 WebGL2，并把结果写入
`artifacts/installation/browser-check.json`。

安装器使用 Playwright 的 `--with-deps`，同时安装 Chromium 所需的系统动态库。
这一步在 Ubuntu 上需要 root 或 sudo。若浏览器已经下载、只报 `libnspr4.so`
等动态库缺失，可只补系统库并重新验证浏览器，不必再跑 CPU 测试：

```bash
scripts/platform/install_browser --deps-only
```

## 两个本地配置

| 配置 | 物理系统 | 时长 |
| --- | --- | --- |
| `configs/platform/town02_strict_itu_single_ma.json`（默认） | 1 个轨道 MA，1 个 UE，2.8 GHz | 1 秒 |
| `configs/platform/town02_strict_itu_64tx_40ue.json`（默认） | 16 个面板，64 个物理 TX 端口，40 个 UE，2.8 GHz | 4 秒 / 401 个采样时刻 |

原三材质配置 `town02_single_ma.json`、`town02_64tx_40ue.json` 也可显式选择。

基站位置为归档 Straight 1 生成器使用的 `[103.211403, -102.730927, 10]` 米。
UE 初始位置采样自该路段 CSV 的原始 `x_b,y_b,z_b` 坐标，地图未缩放或平移。
UE 使用新定义的缓慢、有界五次曲线运动，**不是原车辆轨迹回放**。
地图里已有的车辆保留为静态网格；默认配置没有额外动态 actor。

```bash
WIWO_GPU=-1 .venv-full/bin/python examples/01_single_ma.py --headless
WIWO_GPU=-1 .venv-full/bin/python examples/02_dynamic_6dma.py --headless
.venv-full/bin/python -m wiwo_ma.platform.town02 check
.venv-full/bin/python -m wiwo_ma.platform.town02_materials check
```

这两条示例只验证机械状态，不计算 CSI。新 Town02 配置尚不能沿用 Munich 历史数据的
独立数值验证结论；原生场景加载、静态几何碰撞检查和 GPU 传播需要另行完成。
历史 HDF5 回放继续使用数据内记录的原场景，不能替换地图后当作 Town02 CSI。

原版示例仍可通过显式配置运行：

```bash
python examples/01_single_ma.py --headless --config configs/platform/single_ma_bounded.json
python examples/02_dynamic_6dma.py --headless --config configs/platform/munich_continuous_full64_4s_NOT_authorized.json
```
