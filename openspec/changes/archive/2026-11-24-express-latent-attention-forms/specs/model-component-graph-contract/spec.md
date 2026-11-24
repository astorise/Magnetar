## ADDED Requirements

### Requirement: Model configuration for latent attention, mixed expert layers and expert routing

`model-config` `1.7.0` SHALL add, without changing any earlier function or record: `multi-head-latent-attention` (`option<mla-config>`), `mixture-of-experts-layers` (`option<list<bool>>`, one entry per layer), `expert-routing` (`option<expert-routing-config>`), and a `yarn` case of `rope-frequency-scaling`. The host SHALL keep serving `1.2.0` to `1.6.0`, and SHALL encode each value against the calling Component's own types, so that a Component compiled against an earlier version fails its call instead of receiving a case it cannot represent.

#### Scenario: The values cross the real component boundary
- **WHEN** a Component compiled against `1.7.0` reads the four functions for a checkpoint declaring latent attention, three layers of which two are MoE, sigmoid routing in four groups and YaRN
- **THEN** it reads back every field the Runtime bound, and reads `none` from each for a checkpoint declaring none of them

#### Scenario: A Component predating YaRN
- **WHEN** a Component compiled against `1.4.0` calls `rope-scaling` for a YaRN checkpoint
- **THEN** the call fails

### Requirement: Declared features a graph does not show must be read

When the bound configuration declares latent attention, mixed expert layers or expert routing, the Runtime SHALL refuse a graph built by a Component that did not call the function carrying that feature since the configuration was bound. Every `rope` node SHALL declare exactly the configuration's frequency scaling, YaRN included.

#### Scenario: A Component that predates the feature
- **WHEN** the Qwen Component compiled against `1.5.0` builds graphs for a checkpoint declaring mixed expert layers or expert routing
- **THEN** the Runtime refuses the graphs and names the unread function

#### Scenario: A YaRN checkpoint without YaRN nodes
- **WHEN** a graph for a YaRN checkpoint has a `rope` node without the YaRN attributes, or with other values
- **THEN** the Runtime refuses the graph
