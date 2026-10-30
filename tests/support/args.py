# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""测试侧统一的参数对象构造器。

设计目标（对应任务 No.13 的验收要求）：

- **只有一处默认值来源**：直接复用 ``get_slime_extra_args_provider()`` 注册的真实
  参数定义，``parse_args([])`` 得到的就是声明默认值，测试里不再手抄默认值表。
- **只声明差异**：用例通过 ``build_relax_args(**overrides)`` 覆盖自己关心的字段，
  其余字段自动取声明值，因此不会因为新增参数而失效。
- **暴露而不是掩盖**：覆盖一个未声明的参数名会立刻报错，而不是像以前那样被
  ``getattr(args, ..., default)`` 静默吞掉。
"""

import argparse
import copy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from tests.support.megatron_defaults import MEGATRON_TEST_DEFAULTS


_IMPORT_GUARD_READY = False
_BASE_ARGS: dict[str, Any] | None = None


def _ensure_router_stub() -> None:
    """保证 RouterArgs 具备 add_cli_args 签名。

    其他测试（例如 test_arguments_spec_driven）会把 RouterArgs 替换成 ``object`` 以
    隔离依赖，本构造器复用同一个模块对象，因此需要在校验前补齐可调用的桩。 真实 sglang_router 不受影响。
    """
    module = sys.modules.get("sglang_router.launch_router")
    if module is None:
        return
    router_args = getattr(module, "RouterArgs", None)
    if not hasattr(router_args, "add_cli_args"):
        module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)


def _install_optional_dependency_stubs() -> None:
    """在没有 sglang 可选依赖的环境里让参数定义模块可导入。

    CPU CI 不安装 sglang/sglang_router，而 ``relax.utils.arguments`` 在模块顶层就引用
    它们。这里只补最小桩：参数声明本身由 Relax 提供，桩不影响任何默认值。 真实依赖存在时不做任何替换。
    """
    global _IMPORT_GUARD_READY
    _ensure_router_stub()
    if _IMPORT_GUARD_READY:
        return

    try:
        import sglang_router.launch_router  # noqa: F401
    except ImportError:
        router_pkg = ModuleType("sglang_router")
        launch_router = ModuleType("sglang_router.launch_router")
        # add_cli_args 由 RouterArgs 提供；桩只需满足调用签名。
        launch_router.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
        sys.modules.setdefault("sglang_router", router_pkg)
        sys.modules.setdefault("sglang_router.launch_router", launch_router)

    try:
        import relax.backends.sglang.arguments as sglang_arguments_module
    except ImportError:
        sglang_arguments_module = ModuleType("relax.backends.sglang.arguments")
        sglang_arguments_module.sglang_parse_args = lambda: None
        sglang_arguments_module.validate_args = lambda args: args
        sys.modules["relax.backends.sglang.arguments"] = sglang_arguments_module

    # 其它用例可能把它替换成更简单的桩，这里补齐构造器需要的入口。
    if not hasattr(sglang_arguments_module, "add_sglang_arguments"):
        sglang_arguments_module.add_sglang_arguments = lambda parser: parser
    if not hasattr(sglang_arguments_module, "sglang_parse_args"):
        sglang_arguments_module.sglang_parse_args = lambda: None

    _IMPORT_GUARD_READY = True


def build_relax_argparser() -> argparse.ArgumentParser:
    """构造与生产一致（不含 Megatron 部分）的 Relax 参数解析器。"""
    _install_optional_dependency_stubs()

    from relax.utils import arguments as relax_arguments

    # 其他测试可能把 RouterArgs 换成 object 来隔离依赖；模块级名字在导入时已绑定，
    # 这里就地修正，保证构造器不依赖用例的执行顺序。
    if not hasattr(relax_arguments.RouterArgs, "add_cli_args"):
        relax_arguments.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)

    get_slime_extra_args_provider = relax_arguments.get_slime_extra_args_provider

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    get_slime_extra_args_provider()(parser)
    return parser


def _base_attributes() -> dict[str, Any]:
    """按生产解析顺序得到一份完整属性表：主 parser -> pre-parse -> finalize。"""
    global _BASE_ARGS
    if _BASE_ARGS is not None:
        return _BASE_ARGS

    _install_optional_dependency_stubs()
    from relax.utils.args_finalize import finalize_args
    from relax.utils.arguments import build_pre_parse_parser

    args = build_relax_argparser().parse_args([])
    existing = vars(args)
    for name, value in vars(build_pre_parse_parser().parse_args([])).items():
        existing.setdefault(name, value)
    finalize_args(args)
    _BASE_ARGS = dict(existing)
    return _BASE_ARGS


def declared_parameter_names() -> set[str]:
    """返回所有可用的参数名（唯一来源：参数定义 + finalize 补齐的字段）。"""
    return set(_base_attributes()) | set(MEGATRON_TEST_DEFAULTS)


def declared_parameter_defaults() -> dict[str, Any]:
    """返回声明默认值字典，供一致性测试与前后对照使用。"""
    return copy.deepcopy(_base_attributes())


def build_relax_args(**overrides: Any) -> argparse.Namespace:
    """构造一个字段齐全的参数对象，用例只声明自己关心的差异。

    Args:
        **overrides: 需要覆盖的参数名 -> 取值。参数名必须来自参数定义、
            finalize 补齐的字段，或 ``MEGATRON_TEST_DEFAULTS``；拼写错误会抛
            ``AttributeError``（这是有意的：拒绝把错误静默忽略）。

    Returns:
        与生产解析结果同构的 ``argparse.Namespace``。
    """
    base = copy.deepcopy(_base_attributes())
    known = set(base) | set(MEGATRON_TEST_DEFAULTS)
    # sglang / router 字段由 sglang 子解析器声明；未安装 sglang 的环境（CPU CI）里
    # 构造器拿不到声明，因此按前缀放行，具体取值由用例显式给出。
    unknown = sorted(name for name in set(overrides) - known if not name.startswith(("sglang_", "router_")))
    if unknown:
        raise AttributeError(f"unknown Relax argument(s) {unknown}; declared names come from relax/utils/arguments.py")
    args = argparse.Namespace(**{**copy.deepcopy(MEGATRON_TEST_DEFAULTS), **base})
    for name, value in overrides.items():
        setattr(args, name, value)
    return args
