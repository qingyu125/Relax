# 参数默认值清理：范围、前后对照与剩余兼容项

> 对应贡献者任务 **No.13「参数默认值集中管理与兼容逻辑规范化」**。
> 本文是验收材料：说明清理了哪些地方、改成了什么、还剩什么以及为什么留下。

## 1. 目标与结论

Relax 的参数定义里已经声明了默认值，但业务代码长期用
`getattr(args, "task_type", "causal_lm")` 这种写法再补一遍。后果有三个：

1. 同一个默认值要改两处（甚至 20 处，如 `fully_async`）。
2. 值会漂移：`seed` 声明 1234，8 处兜底却写 42。
3. 字段根本不存在时，`getattr` 把问题伪装成"用了默认值"（本次发现 11 个这样的幽灵属性）。

本次改动把读取方式收敛为 **声明 → 收尾 → 校验** 三段：

| 阶段 | 位置 | 职责 |
| --- | --- | --- |
| 声明 | `relax/utils/arguments.py`、`backends/{megatron,fsdp,sglang}/arguments.py`、`utils/opd/opd_utils.py` | 唯一的默认值来源 |
| 收尾 | `relax/utils/args_finalize.py`（新增） | 补全被跳过的子解析器字段、给运行期派生属性初值 |
| 校验 | 既有 `validate_*` | 直接读 `args.xxx` |

结果：`relax/` 内"自有参数用 getattr 补默认值"从 **806 处降到 1 处**（该处对象不是 args，
见 §4.1），`hasattr(args, ...)` 从 **35 处降到 12 处**（全部是模型 config、Megatron 字段或别名桥接）。

## 2. 清理范围

| 模块 | 改造前 | 改造后 |
| --- | --- | --- |
| `relax/utils` | 284 | 1（`models/kimi_k3/model.py` 的误判，见 §4.1） |
| `relax/backends` | 207 | 0 |
| `relax/engine` | 131 | 0 |
| `relax/components` | 92 | 0 |
| `relax/core` | 30 | 0 |
| `relax/distributed` | 27 | 0 |
| `relax/models` | 14 | 0 |
| `relax/agentic` | 7 | 0 |
| `relax/algorithms` / `entrypoints` | 6 | 0 |
| `examples/`（deepeyes、kimi-k3、genrm） | 8 | 0 |
| **合计** | **806** | **1** |

同时清理的类型：

- `getattr(<args>, "自有参数", ...)`：775 + 107 处，改为 `args.xxx`。
- `hasattr(args, "自有参数") and args.自有参数`：`megatron_peft_utils` 3 处、`multimodal/config.py` 9 处、
  `train_metric_utils` 4 处、`arguments.py` 2 处、`distributed/ray/rollout.py` 1 处。
- `getattr(get_args(), ...)` 与 `getattr(group.args, ...)` 这类同义写法：一并按 `get_args().xxx` /
  `group.args.xxx` 处理。
- `except AttributeError` / `vars(args).get(...)`：仓库本来没有用于参数兜底的用法，本次以静态测试固化。

## 3. 前后对照

### 3.1 直接读取

```diff
- if getattr(args, "task_type", "causal_lm") == "causal_lm":
+ if args.task_type == "causal_lm":
```

### 3.2 值分叉（改造前同参数不同兜底值）

| 参数 | 声明值 | 旧兜底值 | 处理 |
| --- | --- | --- | --- |
| `seed` | `1234` | `42`（8 处） | 统一按声明值 1234 |
| `use_metrics_service` | `True` | `False`（5 处） | 统一按声明值 True |
| `apply_chat_template_kwargs` | `{}` | `None`（8 处） | 统一按声明值 `{}` |
| `scale_in_shutdown_timeout` | `30.0` | `20.0`（2 处） | 统一按声明值 30.0 |
| `opd_kl_coef` | `1.0` | `0.0` | 统一按声明值 1.0 |
| `mtp_loss_scaling_factor` | `0.2` | `0.0` | 统一按声明值 0.2 |
| `preference_max_length` | `1024` | `0` | 统一按声明值 1024 |
| `preference_max_completion_length` | `512` | `0` | 统一按声明值 512 |

> 这些分叉在**生产路径**上不会触发（解析后的 args 一定带该字段），它们只在手写
> `Namespace` 的场景生效——也就是测试。因此本改动不影响线上行为，但会让不完整的
> 测试参数对象立即报错，这正是任务要求暴露的问题。

### 3.3 运行期派生属性

`is_vl_model`、`uses_unsplit_forward`、`sft_image_preprocess_on_rank`、`use_critic`
在解析期无法确定，由运行期组件写入；读取点原先靠 `getattr(..., False)` 兜底。
现在由 `args_finalize.seed_runtime_derived()` 给出初值，读取点直接 `args.xxx`。

### 3.4 测试侧

```diff
- def _args(estimator="grpo", **overrides):
-     base = dict(advantage_estimator=estimator, n_samples_per_prompt=4, ...)  # 14 项手抄
-     base.update(overrides)
-     return SimpleNamespace(**base)
+ def _args(estimator="grpo", **overrides):
+     return build_relax_args(**{"advantage_estimator": estimator, "n_samples_per_prompt": 4, **overrides})
```

`tests/utils/test_arguments_opd_teacher_colocate.py::_opd_args` 是典型收益：
手工维护的 **105 个字段收缩到 18 个真实差异**，其余 87 个自动取自声明。

## 4. 剩余项与理由

### 4.1 唯一保留的 `getattr`（1 处）

| 位置 | 原因 |
| --- | --- |
| `relax/models/kimi_k3/model.py:160` `getattr(config, "vision_dp_when_tp", False)` | 这里的 `config` 是 **HF 模型 config**，不是解析后的 args；`vision_dp_when_tp` 恰好与 Relax 参数同名，属于命名碰撞 |

### 4.2 Megatron 侧兼容（集中到 `relax/backends/megatron/args_compat.py`）

| 类别 | 字段 | 说明 |
| --- | --- | --- |
| 字段改名 | `norm_epsilon` ⇄ `layernorm_epsilon` | `FIELD_ALIASES` 表 + `resolve_aliased_field` |
| 字段缺失/能力 | `dynamic_context_parallel` | `supports_dynamic_context_parallel()` 用 `mpu.initialize_model_parallel` 签名探测，兼容 dev 分支与回移植 |
| 缺省值 | `rope_type` | `set_rope_type_default()`，仅字段存在时写入 |
| 旧版行为 | `initial_loss_scale` / `min_loss_scale` / `use_precision_aware_optimizer` / `store_param_remainders` | `resolve_optimizer_precision_args()`，bf16 路径仍取上游 `OptimizerConfig` 默认值 |

Megatron 官方字段（`context_parallel_size`、`seq_length`、`adam_*`、`fp8*` 等）仍按
`getattr`/直读混用，因为它们由上游 parser 提供、不在 Relax 声明范围内；这些字段一旦被
Relax 业务代码直读，测试就需要提供，见 `tests/support/args.py::MEGATRON_TEST_DEFAULTS`。

### 4.3 保留的 `hasattr`（12 处）

| 位置 | 原因 |
| --- | --- |
| `relax/backends/megatron/args_compat.py` ×3 | 本模块即兼容层白名单 |
| `relax/backends/megatron/model.py:397` | `hasattr(args, field.name)` 批量探测 Megatron dataclass 字段 |
| `relax/backends/megatron/model_provider.py:495` | `gradient_accumulation_fusion` 为 Megatron 字段 |
| `relax/backends/sglang/arguments.py:312` | sglang 上游改名（`dp_size`/`data_parallel_size` 等）的别名桥接 |
| `relax/backends/sglang/sglang_engine.py` ×2 | 动态字段名 `f"sglang_{name}"` |
| `relax/core/node_group_affinity.py:50` | 内部标记属性 `_CONTROL_PLANE_NODE_GROUP_ATTR` |
| `relax/utils/arguments.py:3934` | `--custom-config-path` 的任意 YAML 键，用于"已存在则告警" |
| `relax/utils/arguments.py:4034` | 旧脚本别名 `--enable-gloo-process-groups`（两个名字都不在参数定义里，遗留项） |
| `relax/utils/arguments.py:4075` | `eval_datasets` 是校验期派生字段，按其存在性清理 |
| `relax/backends/megatron/actor.py`、`utils/tq/lifecycle.py`、`utils/training/flops_counter.py` ×5 | 对象分别是模型 config / dict-like config，不是 args |

### 4.4 已知遗留（后续 PR 处理）

1. `--enable-gloo-process-groups` / `use_gloo_process_groups` 两个名字都不在参数定义中，
   属于历史别名：建议迁移到 `args_finalize` 的别名归一化表，或在下一个 minor 版本移除。
2. 11 个幽灵属性（`genrm_ray_num_gpus`、`rollout_request_timeout`、`eviction_check_interval`、
   `fp8_param` 等）仍然只出现在读取点。它们对应的是"从未声明的开关"，需要产品确认是删除还是补声明。
3. `--tool-key` 的声明默认值是 `"tools"`，与 `--sft-training-mode cpt` 的互斥检查冲突，
   即 CPT 必须显式传 `--tool-key ""`。这是既有语义，本次未改，建议单开 issue。

## 5. 验证

| 检查 | 结果 |
| --- | --- |
| `ruff check relax/ examples/ scripts/` | 通过 |
| `ruff format --check` | 通过 |
| `python -m compileall relax examples scripts` | 通过 |
| `pytest tests/algorithms` | 205 passed（与基线一致；9 个失败为缺少 MSVC 导致 `torch.compile` 失败） |
| `pytest tests/utils`（跳过无法收集的模块） | 985 passed；失败项与基线逐条对照后均为环境依赖缺失（见 PR 描述） |
| `pytest tests/utils/test_no_getattr_defaults.py` | 通过（静态守卫） |
| `pytest tests/utils/test_argument_defaults_contract.py` | 通过（声明↔构造器一致性、幂等、边界值） |
| `pytest tests/backends/megatron/test_args_compat.py` | 21 passed / 2 skipped（Megatron 相关用例在真实镜像中生效） |

GPU/多节点路径（Megatron、SGLang、Ray）无法在本机验证，需在官方镜像
`ghcr.io/redai-studio/relaxrl:latest` 里跑全量 `pytest tests/`。
