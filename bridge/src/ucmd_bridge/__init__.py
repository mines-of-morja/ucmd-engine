"""UnitcellMD bridge: external backends for `ucmd run` (PLAN.md E3.7, ADR 0010).

`ucmd-openmm` runs the model of a system package with OpenMM, e.g. on a GPU through OpenCL.
`ucmd run` starts it when the machine profile selects `[backend] kind = "openmm"`, after writing
a reference single point that the backend must reproduce before it runs.
"""

__version__ = "0.1.0.dev0"
