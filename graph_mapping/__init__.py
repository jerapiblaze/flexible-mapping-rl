from gymnasium.envs.registration import register

register(
    id="graph_mapping/static_mapping-v0",
    entry_point="graph_mapping.envs:StaticMappingEnv",
)

register(
    id="graph_mapping/static_mapping-v2",
    entry_point="graph_mapping.envs:StaticMapping2Env",
)
