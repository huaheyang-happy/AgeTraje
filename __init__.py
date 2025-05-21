# -*- coding: utf-8 -*-
"""
scglue.models
"""

from typing import Optional

from anndata import AnnData

from ..utils import config
from .base import Model
# Import original SCGLUE model if it still exists and is needed
# from .glue import GLUE # Example
# from .scglue import SCGLUEModel, configure_dataset # Example

# --- Import the NEW CGLUE-SOE-OT Model ---
# Ensure the file is named cglue_soe_ot.py and located here
try:
    from .cglue_soe_ot import CGLUESOE_OT_Model, configure_dataset_cglue_soe
except ImportError:
    # Provide a fallback or raise a more informative error if the new file isn't found
    print("Warning: CGLUE-SOE-OT model not found in scglue.models.")
    # Optionally define placeholder names if needed elsewhere, or just let it fail later
    CGLUESOE_OT_Model = None
    configure_dataset_cglue_soe = None


# --- Keep original configure_dataset if it exists and is different ---
# Example: Assuming the original configure_dataset was defined here or imported
# def configure_dataset(
#     adata: AnnData,
#     prob_model: str,
#     use_highly_variable: bool = True,
#     use_layer: Optional[str] = None,
#     use_rep: Optional[str] = None,
#     use_batch: Optional[str] = None,
#     use_cell_type: Optional[str] = None,
#     use_dsc_weight: Optional[str] = None,
#     use_obs_names: bool = False
# ) -> None:
#     """
#     Configure dataset for SCGLUE model training
#     (Original configuration function)
#     """
#     # ... original implementation ...
#     pass


__all__ = [
    "Model",
    # "SCGLUEModel", # Uncomment if original SCGLUE is kept
    # "configure_dataset", # Uncomment if original configure_dataset is kept
    "CGLUESOE_OT_Model", # Export the new model
    "configure_dataset_cglue_soe" # Export the config function (can be renamed if desired)
]
