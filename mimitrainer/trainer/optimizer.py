from lion_pytorch import Lion
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

def separate_wd_params(models, wd_layer_keywords, wd_ndim):
    """
    Separates model parameters into those with and without weight decay.
    
    Parameters:
    - models: A list or dictionary of models whose parameters need to be separated.
    - wd_layer_keywords: List of keywords to identify weight decay layers (e.g., Transformer layers).
    - wd_ndim: The minimum number of dimensions for weight decay (e.g., 2 for matrices/tensors).
    
    Returns:
    - wd_params: Parameters that will have weight decay applied.
    - no_wd_params: Parameters that will not have weight decay applied.
    """
    wd_params, no_wd_params = [], []

    # Ensure models is a list even if a single model is passed
    if isinstance(models, dict):
        models = list(models.values())
    elif not isinstance(models, list):
        models = [models]

    for model in models:
        for name, param in model.named_parameters():
            # Check if the parameter belongs to weight-decay layers
            if any(key in name for key in wd_layer_keywords):
                # Weight decay is applied only to parameters with ndim >= wd_ndim
                if param.ndimension() >= wd_ndim:
                    wd_params.append(param)
                else:
                    no_wd_params.append(param)
            else:
                no_wd_params.append(param)

    return wd_params, no_wd_params


def get_optimizer_with_ema(
    models,  # Dictionary or list of discriminators
    lr=8e-4,
    wd=5e-2,
    betas=(0.5, 0.9),
    eps=1e-8,
    use_lion=False,
    ema_decay=0.99,
    wd_module=["encoder_transformer", "decoder_transformer"],
    wd_ndim=2,
    use_ema=True,
):
    """
    Configures the optimizer with weight decay applied only to Transformer parameters
    and sets up Exponential Moving Average (EMA) for all discriminators.

    Parameters:
    - discriminators: A dictionary or list of discriminators.
    - lr: Learning rate for the optimizer.
    - wd: Weight decay value for Transformer parameters.
    - betas: Coefficients for computing running averages of gradient and its square.
    - eps: Term added to the denominator to improve numerical stability.
    - use_lion: If True, uses Lion optimizer instead of AdamW.
    - ema_decay: EMA decay rate (e.g., 0.99).

    Returns:
    - optimizer: Configured optimizer.
    - ema_models: A dictionary or list of EMA models that track the exponential moving average of the weights.
    """
    # Ensure discriminators is a list even if a single model is passed
    if isinstance(models, dict):
        model_names = list(models.keys())
        models = list(models.values())
    elif not isinstance(models, list):
        models = [models]
        model_names = ['model_0']

    # Get all parameters from the discriminators
    wd_params, no_wd_params = separate_wd_params(models, wd_module, wd_ndim)

    # Define parameter groups
    params = [
        {"params": wd_params, "weight_decay": wd},    # Weight decay for Transformer params
        {"params": no_wd_params, "weight_decay": 0}, # No weight decay for others
    ]

    # Select optimizer type
    if use_lion:
        optimizer = Lion(params, lr=lr, betas=betas, weight_decay=wd)
    else:
        optimizer = AdamW(params, lr=lr, betas=betas, eps=eps)

    if use_ema:
        # Set up EMA for all models (discriminators in this case)
        # You need to create a dictionary of averaged models for EMA
        ema_models = {name: AveragedModel(model, avg_fn=lambda avg_param, param: ema_decay * avg_param + (1 - ema_decay) * param)
                      for name, model in zip(model_names, models)}

        return optimizer, ema_models
    else:
        return optimizer
