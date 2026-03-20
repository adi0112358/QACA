from dataclasses import dataclass, field


@dataclass
class DeltaCRDTMap:
    state: dict = field(default_factory=dict)

    def apply_delta(self, key, delta):
        self.state[key] = self.state.get(key, 0) + delta

    def merge(self, other_state):
        for key, value in other_state.items():
            self.state[key] = max(self.state.get(key, 0), value)

    def value(self):
        return dict(self.state)


@dataclass
class ConsensusLog:
    entries: list = field(default_factory=list)

    def append(self, entry):
        self.entries.append(entry)

    def committed(self):
        return list(self.entries)


@dataclass
class VectorClock:
    clock: dict = field(default_factory=dict)

    def tick(self, node_id):
        self.clock[node_id] = self.clock.get(node_id, 0) + 1

    def merge(self, other):
        for key, value in other.clock.items():
            self.clock[key] = max(self.clock.get(key, 0), value)


@dataclass
class CausalLog:
    entries: list = field(default_factory=list)

    def append(self, entry, clock: VectorClock):
        self.entries.append((dict(clock.clock), entry))

    def committed(self):
        return list(self.entries)


@dataclass
class MixedConsistencyLayer:
    overlays: dict = field(default_factory=dict)
    overlay_types: dict = field(default_factory=dict)
    ordered_log: ConsensusLog = field(default_factory=ConsensusLog)
    causal_logs: dict = field(default_factory=dict)

    def register_overlay(self, overlay_id, consistency):
        self.overlay_types[overlay_id] = consistency

    def apply_overlay(self, overlay_id, key, delta):
        overlay = self.overlays.setdefault(overlay_id, DeltaCRDTMap())
        overlay.apply_delta(key, delta)

    def commit_ordered(self, entry):
        self.ordered_log.append(entry)

    def commit_causal(self, overlay_id, entry, node_id="node"):
        log = self.causal_logs.setdefault(overlay_id, CausalLog())
        clock = VectorClock()
        clock.tick(node_id)
        log.append(entry, clock)
