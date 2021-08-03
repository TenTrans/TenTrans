from .mlm_task import MLMTask
from .classification_task import ClassificationTask
from .seq2seq_task import Seq2SeqTask
from .tlm_task import TLMTask
from .unsup_mass_task import UnsuperMassTask

task_builder = {
    'mlm': MLMTask,
    'classification': ClassificationTask,
    'seq2seq': Seq2SeqTask,
    'tlm': TLMTask,
    'unsup_mass': UnsuperMassTask
}
