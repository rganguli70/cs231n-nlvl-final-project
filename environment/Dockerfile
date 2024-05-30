FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch230

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip list