import os
import torch
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import sys



def fairseq_pt_hf_ckpt(fairseq_pt: str):
    '''
    This function takes a fairseq wav2vec2 model and converts it to a hf model.
    It will create the checkpoint that is compatible, but you will need to add the config file yourself.
    You can find an example of config file in the examples/wavlm/utils/hf_config directory.
    '''
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_pt])  # Instantiate the fairseq model
    model = models[0]
    model.eval()

    model.__class__.__name__ = "Wav2Vec2Model"  # Rename the model name to allow conversion to torchaudio
    model = import_fairseq_model(model).eval()  # Convert the fairseq model to torchaudio

    fairseq_dict = model.state_dict()  # Get the torchaudio dictionary

    print(f"Keys of the first dict: {fairseq_dict.keys()}")
    # print(f"Some weights: {fairseq_dict['feature_extractor.conv_layers.0.layer_norm.weight']}")
    hf_dict = map_fairseq_to_hf(fairseq_dict)

    print(f"Keys of the second dict: {hf_dict.keys()}")
    # print(f"Some weights: {hf_dict['feature_extractor.conv_layers.0.layer_norm.weight']}")

    torch.save(hf_dict, (os.path.splitext(fairseq_pt)[0]) + '_hf.ckpt')  # Save the new dictionary

    print((os.path.splitext(fairseq_pt)[0]) + '.ckpt created !')


def map_fairseq_to_hf(fairseq_dict: dict):
    # This function maps the fairseq model to a HF model
    new_map = {}
    wav2vec2_map = fairseq_dict
    print(wav2vec2_map.keys())
    mapping = {
        'encoder.transformer.pos_conv_embed.conv.bias': 'encoder.pos_conv_embed.conv.bias',
        'encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0': 'encoder.pos_conv_embed.conv.weight_g',
        'encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1': 'encoder.pos_conv_embed.conv.weight_v',
        'encoder.transformer.layer_norm.weight': 'encoder.layer_norm.weight',
        'encoder.transformer.layer_norm.bias': 'encoder.layer_norm.bias'
    } # I'm not a fan of this being hard coded but I did not have a better idea

    # We iterate over all keys:
    for key in wav2vec2_map.keys():
        # We deal with the feature projection params
        if 'feature_projection' in key:
            modified_key = key.replace('encoder.', '')
            new_map[modified_key] = wav2vec2_map[key]
        # We deal with some specific layers with bigger changes
        elif key in mapping:
            modified_key = mapping[key]
            new_map[modified_key] = wav2vec2_map[key]
        # We deal with the rest, i.e. the tranformer layers
        elif 'encoder.transformer.layers' in key:
            modified_key = key.replace('encoder.transformer.layers.', 'encoder.layers.')
            new_map[modified_key] = wav2vec2_map[key]
        else: 
            # We deal with the feature extractor
            new_map[key] = wav2vec2_map[key]
    return new_map
    
if __name__ == "__main__":
    fairseq_pt_hf_ckpt(sys.argv[1])