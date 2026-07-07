from .cfc_module import CfCTemporalModule
from .lstm_module import LSTMTemporalModule
from .gru_module import GRUTemporalModule
from .transformer_module import TransformerTemporalModule

TEMPORAL_MODULES = {
    'cfc': CfCTemporalModule,
    'lstm': LSTMTemporalModule,
    'gru': GRUTemporalModule,
    'transformer': TransformerTemporalModule,
    'none': None,
}
