# The pre-trained model

Due to the file size limit of GitHub, I have uploaded the pre-trained model to Google Drive. You can download the pre-trained model from the following link:

[Pre-trained model](https://drive.google.com/file/d/1FTnZVdOhMe4Kl5ZiIVwszRtOBvzxGWx3/view?usp=sharing)

Or you can also use the following command to download the model:

```bash
gdown --id 1FTnZVdOhMe4Kl5ZiIVwszRtOBvzxGWx3
```

And here is the Python code to download the model:

```python
import gdown
url = 'https://drive.google.com/uc?id=1FTnZVdOhMe4Kl5ZiIVwszRtOBvzxGWx3'
output = 'model.pth'
gdown.download(url, output, quiet=False)
```

> Note that this is a quantized model so you need to add the parameter `quantized=True` when using the inference script.

The original model is too large, and if you want to request a copy of that, please feel free to contact me. [contact](../readme.md#contact)