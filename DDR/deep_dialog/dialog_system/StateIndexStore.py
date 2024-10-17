class StateIndexStore:
    def __init__(self):
        self.state_index_map = {}

    def add_state(self, state, index):
        self.state_index_map[state] = index

    def get_index(self, state, state_index_store):
        self.state_index_map = state_index_store
        return self.state_index_map.get(state)

    def has_state(self, state, state_index_store):
        self.state_index_map = state_index_store
        return state in self.state_index_map