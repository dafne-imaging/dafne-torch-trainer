from typing import Callable, Iterable, List, Dict, Any
from .events import EngineEvents
from .state import EngineState

class Engine():
    def __init__(self, process_function: Callable[['Engine', Any], Any]):

        self.process_function = process_function
        self.state = EngineState()
        self._handlers: Dict[EngineEvents, List[tuple]] = {event: [] for event in EngineEvents}
    
    def add_event_handler(self, event: EngineEvents, handler: Callable, *args, **kwargs):
        '''
        Add event handler to the engine
        '''
        if event not in EngineEvents:
            raise ValueError(f"Event {event} not found in EngineEvents")
        self._handlers[event].append((handler, args, kwargs))
    
    def fire_event(self, event_name:EngineEvents):
        '''
        Fire event to the engine
        '''
        for handler, args, kwargs in self._handlers[event_name]:
            handler(self, *args, **kwargs)
    
    def run(self, data_loader: Iterable, max_epochs: int, check_stop=None):
        '''
        Start training loop
        '''
        self.state.max_epochs = max_epochs
        self.state.terminated = False
        self.state.iteration = 0
        self.state.check_stop = check_stop

        self.fire_event(EngineEvents.STARTED)

        for epoch in range(max_epochs):
            if check_stop and check_stop():
                break
            self.state.epoch = epoch + 1
            self.fire_event(EngineEvents.EPOCH_STARTED)
            epoch_loss = 0.0
            batch_count = 0

            for batch in data_loader:
                if check_stop and check_stop():
                    break
                self.state.iteration += 1
                self.state.batch = batch
                self.fire_event(EngineEvents.ITERATION_STARTED)
                
                self.state.output = self.process_function(self, batch)
                
                if isinstance(self.state.output, dict) and 'loss' in self.state.output:
                    self.state.loss = self.state.output['loss']
                    epoch_loss += self.state.output['loss']
                    batch_count += 1
                
                self.fire_event(EngineEvents.ITERATION_COMPLETED)
                
            if batch_count > 0: 
                self.state.metrics['avg_loss'] = epoch_loss / batch_count
            
            self.fire_event(EngineEvents.EPOCH_COMPLETED)
        
            if self.state.terminated:
                break
        
        self.fire_event(EngineEvents.COMPLETED)
        return self.state