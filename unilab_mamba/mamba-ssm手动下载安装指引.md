# mamba-ssm 手动下载与安装指引

> 背景:UniLab 机器装 mamba-ssm 卡在 github 大文件(533MB)下载不稳。
> causal-conv1d 已装好(193MB,已下完装上)。只差 mamba-ssm。
> 由你来手动下,下完通知 Claude 接验证 + 续训。

---

## 一、要下的文件

**mamba-ssm 预编译 wheel(533MB,匹配本机 torch2.7+cu128 / cp313)**

下载链接:
```
https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

下载到(文件名必须完整规范,uv 才认):
```
/tmp/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

目标大小:533310725 字节(约 533MB)

---

## 二、下载方法(在 UniLab 机器上,tmux 里跑)

这台机器直连 github 不稳(下到 100-160MB 会断),必须先开 autodl 加速:

### 方法 A:断点续传(推荐,接着已下的 140MB)

已有 140MB 在 `/tmp/mamba_ssm.whl`,先改成规范名,再 `-c` 续传:

```bash
source /etc/network_turbo
cp /tmp/mamba_ssm.whl "/tmp/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
cd /tmp
wget -c "https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
```

### 方法 B:从头下(如果方法 A 续传有问题)

```bash
source /etc/network_turbo
cd /tmp
rm -f mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
wget "https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
```

### 如果 wget 也下不稳(断断续续),用循环续传:

```bash
source /etc/network_turbo
cd /tmp
URL="https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
for i in $(seq 1 40); do
  sz=$(stat -c%s "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" 2>/dev/null || echo 0)
  [ "$sz" = "533310725" ] && echo "DONE $sz" && break
  echo "i=$i size=$sz"
  curl -L -C - --connect-timeout 15 --max-time 180 -sS "$URL" -o "mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" 2>/dev/null
done
```

建议在 tmux 里跑(防 ssh 断):`tmux new -s dl` 然后执行上面的命令。

---

## 三、验证下载完整

```bash
ls -la /tmp/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
# 大小应为 533310725
stat -c%s /tmp/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

---

## 四、安装命令(下完执行,或交给 Claude)

```bash
cd /root/UniLab_WS/UniLab
export PATH=/root/miniconda3/envs/unilab/bin:$PATH
export VIRTUAL_ENV=/root/UniLab_WS/UniLab/.venv
uv pip install "/tmp/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
```

验证:
```bash
.venv/bin/python -c "import mamba_ssm; print('MAMBA_OK', mamba_ssm.__version__)"
.venv/bin/python -c "from unilab.algos.torch.flash_sac.mamba_actor import _HAS_OFFICIAL_MAMBA; print('HAS_OFFICIAL', _HAS_OFFICIAL_MAMBA)"
```

期望输出:
- `MAMBA_OK 2.2.6.post3`(或类似版本号)
- `HAS_OFFICIAL True`(这个为 True 说明 UniLab 源码会自动切官方 CUDA kernel)

---

## 五、装好后的下一步(交给 Claude)

mamba-ssm 装好后,Claude 会:
1. 确认 `_HAS_OFFICIAL_MAMBA=True`(源码自动切官方 kernel,不经 inductor)
2. 用原 D v3 配置(use_compile:true)从 model_4000.pt 续训
3. 监控能否过 940/980 这个坎(之前 compile recompile 死锁点)
4. 若过了 → 根治验证成功,D v3 卡死根因确认是纯 PyTorch SSM 触发 recompile,官方 kernel 解决

---

## 当前根治进度(2026-06-20 更新)

- ✅ causal-conv1d 已装好(1.6.2.post1)
- ✅ mamba-ssm 2.2.6 装好(533MB wheel,本机循环续传下完→scp 传云机→uv 装本地 wheel)
- ✅ `_HAS_OFFICIAL_MAMBA = True`(切官方 CUDA kernel,不经 inductor → 绕开 recompile 死锁)
- ✅ patch `mamba_ssm/utils/generation.py`(transformers 5.12 兼容,try-except 容错 GreedySearch 旧类,UniLab 不用 generation 不影响)
- ✅ patch `src/unilab/algos/torch/offpolicy/worker.py`(collector actor 上 cuda + obs/dones/priv_info to device + actions.cpu().numpy)—— official kernel 要 cuda,UniLab collector 默认 cpu,已改
- ✅ 前台 12 秒测试:不崩、无 CRASH/Error,patch 正确训练能跑
- ⏳ **唯一卡住:ssh 远程启动长后台训练不可靠**(tmux/nohup/setsid/disown 全"启动即死",云机 sshd/sshpass 硬限制)→ **必须本机终端启动**
- D v3 卡死根因已确认:torch.compile + 纯 PyTorch Mamba SSM 的 recompile 死锁(非 checkpoint bug,非磁盘满),official kernel 是解法

---

## 五点五、关键决策:方案C —— 关 use_compile + 用 official kernel + collector 上 cuda

**方案B(关 compile + 纯 PyTorch SSM)验证结果**:不死锁(collector 在跑、无僵尸),但**太慢** —— 纯 PyTorch SSM collector 推理(4096 envs × for-loop)慢到 10 分钟没出 iter1。证明死锁根因是 compile,但纯 PyTorch 跑不到 940。故转方案C。

### 方案C 配置(已就位)

1. `mamba_phaseD_v3.yaml` `use_compile: false`(L34)—— 不编译 learner actor/critic 前向,**无 recompile 死锁**
2. `mamba_actor.py` `use_official: bool = True`(L114)—— 用 official CUDA kernel,**SSM 快**(collector + learner 都用)
3. `worker.py` patch(collector actor 上 cuda)—— official kernel 要 cuda,collector 默认 cpu 会崩,故改:
   - L341 build_actor device `"cpu"` → `"cuda"`(collector actor 上 GPU)
   - L423/424 `obs_torch/dones_torch` 加 `.to(next(actor.parameters()).device)`
   - L431 `priv_info_torch` 加 `.to(next(actor.parameters()).device)`
   - L439 `actions_torch.numpy()` → `actions_torch.cpu().numpy()`(action 回 cpu 给 env)

### 方案C 原理

- **关 use_compile**:learner 不编译 actor 前向 → SSM 处不 recompile → 无死锁
- **用 official kernel**:SSM 走预编译 CUDA(不经 inductor,不需 torch.compile),快
- **collector 上 cuda**:official kernel 要 cuda 输入,collector actor 必须在 GPU(collector 推理也在 GPU,与 learner 共享 GPU)

### torch.compile 编译什么(背景)

`learner.py:271 _compile_training_methods` 显示 torch.compile **只编译 learner 的三个方法**:`actor.get_mean_and_std`(含 SSM)、`_critic_loss_tensors`、`_actor_loss_tensors`。**collector 不被编译**(独立进程)。D v3 卡死 = learner 编译 SSM 处 recompile 死锁。关 use_compile 即解。

### 已改配置确认

```bash
grep use_compile conf/offpolicy/task/flashsac/g1_multiskill/mamba_phaseD_v3.yaml        # use_compile: false
grep "use_official: bool" src/unilab/algos/torch/flash_sac/mamba_actor.py               # use_official: bool = True
grep -nE "\"cuda\"|to\(next\(actor" src/unilab/algos/torch/offpolicy/worker.py | head    # 5处patch
.venv/bin/python -m py_compile src/unilab/algos/torch/offpolicy/worker.py                # 语法OK
```

### 启动后关键判断(区分慢跑 vs 真卡)

- **collector CPU 高(13核满)+ 快速出 iter(1-2分钟)** → official 生效,方案C 成,盯过 940/980
- **collector CPU 低 + 不出 iter** → 真卡(cuda/同步死锁),需深挖 collector 架构
- **collector CPU 高但不出 iter** → official 没生效或 collector 推理仍慢,检查 `_HAS_OFFICIAL_MAMBA=True`

---

## 六、启动训练(必须在本机/云机终端跑,不能靠 ssh 后台)

**为什么必须本机终端**:通过 ssh(`sshpass ... 'nohup ... &'`)启动的后台任务,ssh 断开时进程被杀(tmux/nohup/setsid/disown 都试过,全失败)。这是云机 sshd 配置 + sshpass 的硬限制。**在云机 web 终端或你的 ssh 客户端里直接跑才稳。**

### 启动命令(在云机终端,tmux 里跑)

```bash
tmux attach -t train 2>/dev/null || tmux new -s train
cd /root/UniLab_WS/UniLab
export PATH=/root/miniconda3/envs/unilab/bin:$PATH
uv run scripts/train_offpolicy.py task=flashsac/g1_multiskill/mamba_phaseD_v3 algo=flashsac 2>&1 | tee /tmp/dv3_train.log
```

- `tmux attach -t train` 复用旧会话(若被远程杀过进程,会话可能还在);不在则新建
- tmux 里跑(Ctrl+B 然后 D 脱离,训练继续;`tmux attach -t train` 重新进入看进度)
- `tee /tmp/dv3_train.log` 同时存日志(便于 Claude 远程读判断)
- **从零训**(不带 `algo.load_run`,因 model_4000.pt 是旧结构,与 6/18 改过的代码不匹配;从零训避开 checkpoint 兼容坑,专验证"过 940/980"根治)
- **方案C配置已就绪**:`use_compile: false` + `use_official: True` + worker.py patch(collector 上 cuda)(见五点五节)

### 启动后预期(方案C:不编译 + official kernel + collector cuda)

1. 前 30 秒-2 分钟:env 创建 + collector 起 + buffer 填充(learning_starts=49)。**不编译**(use_compile=false)
2. official kernel 快,collector 推理快,**1-2 分钟出第一个 iter**(不像方案B 10分钟没出)
3. 出现 `FastFLASHSAC | G1MultiSkill | iter N/8000` 行,iter 推进
4. GPU 利用率高(collector + learner 都在 GPU)+ CPU 多核(MuJoCo 仿真)
5. **目标:过 940/980**(验证死锁根治 + official kernel 可用)

### 启动后关键判断(区分慢跑 vs 真卡)

- **collector CPU 高 + 1-2分钟出 iter** → official 生效,方案C 成,盯过 940/980
- **collector CPU 低 + 不出 iter** → 真卡(cuda/同步死锁),贴日志给 Claude 深挖
- **collector CPU 高但不出 iter** → official 没生效,检查 `_HAS_OFFICIAL_MAMBA=True`
- 报 `GreedySearchDecoderOnlyOutput`:generation.py patch 没生效,检查 try-except
- 其它报错:贴日志给 Claude

---

## 七、根治验证标准(过 940/980 = 成功)

**之前 D v3 反复冻在 980、phaseF 冻在 940**(同一坎:torch.compile recompile 死锁)。这次 official kernel 不经 inductor,不应再触发 recompile。

**验证成功标志**(任一即可确认根治):
- iter 平稳过 980 且继续推进(1000、1500...)→ **recompile 死锁已根治**
- 过 1000 iter 时 save checkpoint 不卡(之前 save 附近冻过,现磁盘已清)

**验证失败(仍卡)**:
- 若又冻在 940-1000 且 GPU 恒 0% + 有僵尸 compile_worker → recompile 死锁未根治(official kernel 没真正生效,检查 `_HAS_OFFICIAL_MAMBA=True`)
- 若冻在别处 → 另有原因,贴日志给 Claude

### 判断卡死的正确方法(避免误判)

- **真卡死**:GPU 连续 ≥5 次采样恒 0%(不是 0-99 波动)+ 进程 CPU TIME 不增 + tensorboard events 停写 + 有僵尸 compile_worker
- **正常训练**:GPU 在 0-99 间规律波动(collector/learner 交替),单次采 0% 是谷值不是卡死
- 用 `nvtop`(持续刷新)比单次 `nvidia-smi` 可靠

---

## 八、相关 memory(已沉淀)

- `mamba-locomotion-empirical-lessons` 教训6:recompile 死锁根因 + 症状鉴别 + 解法
- `autodl-network-turbo-and-ssh-background`:turbo 加速 github + ssh 后台任务不可靠 + causal-conv1d/mamba-ssm 装包要点
- `gpu-stuck-misdiagnosis-continuous-sampling`:卡死判断三件套(连续采样+CPU增长+events写入)

