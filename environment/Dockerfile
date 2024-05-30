FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch230
# RUN pip uninstall torch torchvision torchaudio -y && \
#     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# RUN pip uninstall onnxruntime-training -y && \
#     pip install onnxruntime-training --pre -f https://download.onnxruntime.ai/onnxruntime_nightly_cu118.html && \
#     python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip list