from .cfc_module import CfCTemporalModule
from .lstm_module import LSTMTemporalModule
from .gru_module import GRUTemporalModule

TEMPORAL_MODULES = {
    'cfc': CfCTemporalModule,
    'lstm': LSTMTemporalModule,
    'gru': GRUTemporalModule,
    'none': None,
}
