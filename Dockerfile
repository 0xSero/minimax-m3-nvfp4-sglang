FROM lmsysorg/sglang:dev-cu13-minimax-m3

COPY scripts/patch_loader.py /opt/minimax-m3/patch_loader.py
RUN python3 /opt/minimax-m3/patch_loader.py \
  /sgl-workspace/sglang/python/sglang/srt/models/minimax_m3_vl.py

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["python", "-m", "sglang.launch_server"]

