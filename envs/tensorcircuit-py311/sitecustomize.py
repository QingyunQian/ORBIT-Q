"""Backport TensorCircuit-NG's OMECo contractor shortcut for expert sources.

Source: tensorcircuit/tensorcircuit-ng ``tensorcircuit/cons.py`` at commit
53a712b517cdcaba69ca6376d9d68cd140bdeaea.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import omeco
import tensorcircuit
import tensorcircuit.cons


_original_set_contractor = tensorcircuit.cons.set_contractor


def _set_contractor(
    method: Optional[str] = None,
    optimizer: Optional[Any] = None,
    memory_limit: Optional[int] = None,
    opt_conf: Optional[Dict[str, Any]] = None,
    set_global: bool = True,
    contraction_info: bool = False,
    debug_level: int = 0,
    use_primitives: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    if method and method.startswith("omeco"):
        ntrials = 16
        niters = 32
        if method != "omeco":
            try:
                _, ntrials_text, niters_text = method.split("-")
                ntrials = int(ntrials_text)
                niters = int(niters_text)
            except ValueError as exc:
                raise ValueError(
                    "OMECO contractor shortcut must be 'omeco' or "
                    "'omeco-<ntrials>-<niters>'."
                ) from exc
        method = "custom"
        score = omeco.ScoreFunction(
            tc_weight=1.0,
            sc_weight=0.0,
            rw_weight=64.0,
            sc_target=20.0,
        )
        optimizer = omeco.TreeSA(
            ntrials=ntrials,
            niters=niters,
            score=score,
        )
        kwargs.setdefault("preprocessing", True)

    return _original_set_contractor(
        method=method,
        optimizer=optimizer,
        memory_limit=memory_limit,
        opt_conf=opt_conf,
        set_global=set_global,
        contraction_info=contraction_info,
        debug_level=debug_level,
        use_primitives=use_primitives,
        **kwargs,
    )


tensorcircuit.cons.set_contractor = _set_contractor
tensorcircuit.set_contractor = _set_contractor
