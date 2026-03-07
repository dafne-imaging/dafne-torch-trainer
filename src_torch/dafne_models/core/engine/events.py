from enum import Enum

class EngineEvents(Enum):
    STARTED = 'started'
    EPOCH_STARTED = 'epoch_started'
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'
    EPOCH_COMPLETED = 'epoch_completed'
    COMPLETED = 'completed'