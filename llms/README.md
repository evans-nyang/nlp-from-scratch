# Open Source LLM

## HuggingFace

The models will be leveraged from the HuggingFace library. The library provides a wide range of models that can be used for various NLP tasks. The library also provides a simple API that can be used to load the models and use them for various tasks.

You'll need to set up the HuggingFace Account and create a new authentication token to be able to use the models. The token can be generated from the [HuggingFace website.](https://huggingface.co/settings/tokens)

## Wandb

The `wandb` library will be used to log the training process and visualize the training metrics. The library provides a simple API that can be used to log the training process and visualize the metrics in real-time.

We'll use wandb to fine-tune the models. You can create an account on the [Wandb website.](https://wandb.ai/) and use the API token to authenticate the wandb library in your code.

To obtain a wandb API token, you can create a new project on the wandb website and generate an API token from the project [settings page](https://wandb.ai/settings).

## Models

The models that will be used in this notebook are:

1. ***stablelm-base-alpha-3b-v2*** : `StableLM-Base-Alpha-3B-v2` is a 3 billion parameter decoder-only language model pre-trained on diverse English datasets. This model is the successor to the first `StableLM-Base-Alpha-3B` model, addressing previous shortcomings through the use of improved data sources and mixture ratios.

2. ***orca_mini_3b*** : An `OpenLLaMa-3B` model model trained on explain tuned datasets, created using Instructions and Input from WizardLM, Alpaca & Dolly-V2 datasets and applying Orca Research Paper dataset construction approaches.

3. ***gpt4all*** : `GPT4All` is optimized to run LLMs in the 3-13B parameter range on consumer-grade hardware.LLMs are downloaded to your device so you can run them locally and privately.

## Issues

If you do not have sufficient GPU in your system then the training process might encounter the error below when loading the pretrained model:

```sh
UserWarning: Current model requires 100 bytes of buffer for offloaded layers, which seems does not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using offload_buffers=True.
  warnings.warn(

Some parameters are on the meta device because they were offloaded to the cpu and disk.
```

The log message suggests that your GPU does not have enough memory, so some model parameters were automatically offloaded to the CPU.
The model is likely partially on CPU and partially on GPU, causing device mismatch errors when generating text as shown below.

```sh
UserWarning: You are calling .generate() with the input_ids being on a device type different than your model's device. input_ids is on cuda, whereas the model is on cpu. You may experience unexpected behaviors or slower generation. Please make sure that you have put input_ids to the correct device by calling for example input_ids = input_ids.to('cpu') before running .generate().
  warnings.warn(
```

The error message above is saying that your input_ids (the tokens you passed to the model) are on CUDA (GPU), while your model is on CPU. This device mismatch can cause performance issues or errors.

Consider adding `offload_buffers=True` to your code in the notebook to resolve the issue:

```python
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", offload_buffers=True
)
```

Also, because some layers of the model were offloaded to the CPU or disk due to insufficient GPU memory. The `accelerate` library automatically offloads layers when using `device_map="auto"`.

Below are some of the solutions:

1. Since `accelerate` has already managed offloading for you, you don’t need to call `model.to(device)`. Instead, make sure input tokens match the model's device:

    ```python
    device = model.device  # Get model's current device
    tokens = tokens.to(device)  # Move input tokens to the same device
    ```

    Now, when we call `.generate()`, it should work without errors.

2. If your GPU does not have enough memory, you can force the entire model to run on CPU:

    ```python
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    tokens = tokens.to("cpu")  # Ensure input is on CPU
    ```

    **⚠ Downside**: Running on CPU will be much slower than on GPU!

3. Free GPU Memory & Reduce Model Size

    - Free Up GPU Memory. Before running the model, check and free memory.

        ```python
        import torch
        torch.cuda.empty_cache()
        ```

    - Use 8-bit or 4-bit Quantization (Reduces memory usage significantly).

        ```python
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True  # or load_in_4bit=True
        )

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )

        ```

    - Reduce `generate_len` & Batch Size. Instead of `generate_len=512`, try smaller values like `generate_len=128` to fit in memory.

## Guides

- [stablelm-base-alpha-3b-v2](https://huggingface.co/stabilityai/stablelm-base-alpha-3b-v2)
- [orca_mini_3b](https://huggingface.co/pankajmathur/orca_mini_3b)
- [gpt4all](https://docs.gpt4all.io/)
