# Configs

This folder stores the configs for the transcoder sets that you can use to perform attribution. These are YAML files containing:

- `model_name`: The name of the model that these transcoders were trained on, e.g. "meta-llama/Llama-3.2-1B"
- `feature_input_hook`: The name of the TransformerLens hook point that the transcoders read from, e.g. "hook_resid_mid"
- `feature_output_hook`: The name of the TransformerLens hook point that the transcoders write to, e.g. 'mlp.hook_out'
- `transcoders`: A list of transcoders, each of which is a dictionary containing:
    - `id`: The ID that you assigned to this transcoder, e.g. "Llama-3.2-131k-relu-0"
    - `layer`: The layer that this transcoder was trained on, e.g. 0
    - `filepath`: A local filepath to the transcoder or a URI corresponding to a HuggingFace repo. This should be formatted like "hf://[REPO]/[FILENAME]?revision=[REVISION]", where the revision component is optional. For example, "hf://mntss/skip-transcoder-Llama-3.2-1B-131k-nobos/layer_0.safetensors?revision=new-training"
