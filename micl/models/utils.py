from peft import LoraConfig, get_peft_model, TaskType


def apply_lora(model, r=8, alpha=32, dropout=0.1):
    lora_cfg = LoraConfig(
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    # # freeze all except LoRA
    # for n, p in model.named_parameters():
    #     if "lora_" not in n:
    #         p.requires_grad = False
    return model
