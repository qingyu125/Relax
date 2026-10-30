# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""静态守卫：Relax 自有参数不得再用兜底写法读默认值。

任务 No.13 的验收红线：
- 不通过 ``getattr(args, "own_param", default)`` 重复维护默认值；
- 不通过 ``hasattr(args, "own_param")``、``vars(args).get()``、捕获
  ``AttributeError`` 或通用包装隐藏自有参数缺失。

Megatron 字段差异只允许出现在 ``relax/backends/megatron/args_compat.py``。
"""

import ast
from pathlib import Path

import pytest

from relax.utils.args_finalize import RUNTIME_DERIVED_DEFAULTS
from tests.support.args import build_relax_argparser


REPO_ROOT = Path(__file__).resolve().parents[2]
RELAX_ROOT = REPO_ROOT / "relax"

# args 的等价写法：顶层 args、self.args、group.args 之类，以及 get_args()。
ARGS_OBJECTS = {"args", "self.args", "cfg", "self.cfg", "train_args", "rollout_args"}
# config/self.config 在组件层就是 args，但 relax/models 下同名变量是模型 config。
CONFIG_OBJECTS = {"config", "self.config"}
CONFIG_SAFE_PREFIXES = ("relax/models/",)

# 允许保留 hasattr 的位置：不是"补默认值"，而是删除/别名/上游字段探测。
ALLOWED_HASATTR = {
    ("relax/utils/arguments.py", "eval_datasets"): "eval_datasets 是校验阶段派生字段，这里按存在性清理",
    ("relax/utils/arguments.py", "use_gloo_process_groups"): "旧脚本遗留别名桥接，见 slime_validate_args 注释",
}


def _is_args_like(repo_path: str, obj: str) -> bool:
    if obj in ARGS_OBJECTS or obj.endswith(".args") or obj == "get_args()":
        return True
    return obj in CONFIG_OBJECTS and not repo_path.startswith(CONFIG_SAFE_PREFIXES)


def _own_parameters() -> set[str]:
    """Relax 自有参数的声明名集合（含 pre-parse 与运行期派生属性）。

    注意不要用 tests.support.args.declared_parameter_names()：它额外包含
    Megatron 字段镜像（seq_length、profile_ranks 等），那些字段不属于 Relax 声明，
    允许按 getattr 读取。
    """
    # 先借构造器安装可选依赖桩，再导入参数定义模块。
    parser = build_relax_argparser()
    from relax.utils.arguments import build_pre_parse_parser

    names = {action.dest for action in parser._actions if action.dest != "help"}
    names |= {action.dest for action in build_pre_parse_parser()._actions if action.dest != "help"}
    names |= set(RUNTIME_DERIVED_DEFAULTS)
    return names


def _iter_calls():
    for path in sorted(RELAX_ROOT.rglob("*.py")):
        repo_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"getattr", "hasattr"} and len(node.args) >= 2:
                    key = node.args[1]
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        yield repo_path, node, node.func.id, ast.unparse(node.args[0]), key.value


def test_no_getattr_default_for_relax_parameters():
    own = _own_parameters()
    offenders = [
        f"{repo_path}:{node.lineno} getattr({obj}, {attr!r})"
        for repo_path, node, func, obj, attr in _iter_calls()
        if func == "getattr" and attr in own and _is_args_like(repo_path, obj)
    ]
    assert not offenders, "自有参数必须直接读取 args.xxx：\n" + "\n".join(offenders)


def test_no_hasattr_probe_for_relax_parameters():
    own = _own_parameters()
    offenders = [
        f"{repo_path}:{node.lineno} hasattr({obj}, {attr!r})"
        for repo_path, node, func, obj, attr in _iter_calls()
        if func == "hasattr"
        and attr in own
        and _is_args_like(repo_path, obj)
        and (repo_path, attr) not in ALLOWED_HASATTR
    ]
    assert not offenders, "不要用 hasattr 隐藏自有参数缺失：\n" + "\n".join(offenders)


def test_no_vars_args_get():
    """``vars(args).get()`` 会把缺失参数变成 None，属于被禁止的兜底写法。

    注意只针对 args 对象：``vars(Envs)``、``vars(module)`` 之类的用法与参数无关。
    """
    offenders = []
    for path in sorted(RELAX_ROOT.rglob("*.py")):
        repo_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "vars"
                and node.func.value.args
            ):
                continue
            if _is_args_like(repo_path, ast.unparse(node.func.value.args[0])):
                offenders.append(f"{repo_path}:{node.lineno}")
    assert not offenders, "不要用 vars(args).get() 兜底：\n" + "\n".join(offenders)


def test_megatron_field_probing_stays_in_args_compat():
    """Megatron 字段探测只允许集中在一个文件。"""
    allowed = {"relax/backends/megatron/args_compat.py"}
    offenders = []
    for repo_path, node, func, obj, attr in _iter_calls():
        if func != "hasattr" or repo_path in allowed:
            continue
        if not _is_args_like(repo_path, obj):
            continue
        # 只关心 Megatron 名称，用 args_compat 的别名表判断，避免把业务字段算进来。
        from relax.backends.megatron.args_compat import FIELD_ALIASES

        if attr in FIELD_ALIASES:
            offenders.append(f"{repo_path}:{node.lineno} hasattr({obj}, {attr!r})")
    assert not offenders, "Megatron 字段差异应集中到 args_compat.py：\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", ["relax/utils/args_finalize.py", "relax/backends/megatron/args_compat.py"])
def test_new_modules_exist(path):
    assert (REPO_ROOT / path).is_file()
