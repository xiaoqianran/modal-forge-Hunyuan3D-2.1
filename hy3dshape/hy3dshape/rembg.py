# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image


class BackgroundRemover():
    def __init__(self):
        from rembg import new_session
        self.session = new_session()

    def __call__(self, image: Image.Image):
        from rembg import remove
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output


def prepare_foreground(image: Image.Image, remover=None, force: bool = False,
                       mask: Image.Image = None) -> Image.Image:
    """保留已有透明背景；无透明像素时自动抠图，force 强制重新抠图。"""
    rgba = image.convert("RGBA")
    if mask is not None:
        from PIL import ImageChops
        if mask.size != rgba.size:
            raise ValueError("主体 mask 尺寸必须与原图一致（白色保留，黑色去除）")
        rgba.putalpha(ImageChops.darker(rgba.getchannel("A"), mask.convert("L")))
        return rgba
    if force or rgba.getchannel("A").getextrema()[0] == 255:
        if remover is None:
            remover = BackgroundRemover()
        return remover(image.convert("RGB")).convert("RGBA")
    return rgba


class RMBG2Remover:
    """Optional PR #155-style 1024px segmentation; no silent rembg fallback.

    Uses the model owner's Hugging Face distribution instead of requiring a
    second model hub SDK. Access/weights must already be available to the caller.
    """
    def __init__(self, model_path="briaai/RMBG-2.0", device="cuda", revision=None):
        import torch
        from transformers import AutoModelForImageSegmentation
        self.device = device
        if revision is None and model_path == "briaai/RMBG-2.0":
            revision = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_path, trust_remote_code=True, revision=revision,
            ).eval().to(device)
        except (OSError, ImportError) as error:
            raise RuntimeError(
                "RMBG-2.0 加载失败：请配置已获访问权限的权重或本地模型路径；未自动退回其他抠图器"
            ) from error

    def __call__(self, image):
        import torch
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        rgb = image.convert("RGB")
        with torch.inference_mode():
            prediction = self.model(transform(rgb).unsqueeze(0).to(self.device))[-1].sigmoid()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("RMBG-2.0 输出包含 NaN/Inf")
        alpha = transforms.functional.to_pil_image(prediction[0].float().cpu().squeeze(0))
        result = rgb.convert("RGBA")
        result.putalpha(alpha.resize(rgb.size, Image.Resampling.LANCZOS))
        return result
