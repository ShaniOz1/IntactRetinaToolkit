"""
IntactRetinaToolkit.dataobj
============================
Public API for loading recordings and working with channels.

Usage
-----
    from IntactRetinaToolkit.dataobj import load_rhs, load_edf
    from IntactRetinaToolkit.dataobj import RetinalRecording
    from IntactRetinaToolkit.dataobj import mea_convert_channel, intan_name_to_location
"""

from dataobj.recording import RetinalRecording

from dataobj.rhs_loader import load_rhs
from dataobj.edf_loader import load_edf

from dataobj.channel_utils import (
    mea_name_to_location,
    mea_channel_names_to_locations,
    mea_convert_channel,
    intan_get_locations,
)

__all__ = [
    'RetinalRecording',
    'load_rhs',
    'load_edf',
    'mea_name_to_location',
    'mea_channel_names_to_locations',
    'mea_convert_channel',
    'intan_get_locations',
]