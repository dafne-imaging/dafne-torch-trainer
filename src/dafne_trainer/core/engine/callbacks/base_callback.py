from abc import ABC

class BaseCallback(ABC):
    '''
    Base class for callbacks.
    Provides empty implementations for all engine events.
    Subclasses only need to override the methods they use.
    '''

    def on_started(self, engine):
        pass

    def on_epoch_started(self, engine):
        pass

    def on_epoch_completed(self, engine):
        pass

    def on_iteration_started(self, engine):
        pass

    def on_iteration_completed(self, engine):
        pass

    def on_completed(self, engine):
        pass